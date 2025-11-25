"""LangGraph implementation for hybrid RAG + SQL agent."""
from typing import TypedDict, List, Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, END
import json
import traceback
import dspy

from agent.dspy_signatures import Router, NLToSQL, Synthesizer
from agent.rag.retrieval import TFIDFRetriever
from agent.tools.sqlite_tool import SQLiteTool


class AgentState(TypedDict):
    """State for the agent graph."""
    question: str
    format_hint: str
    route: Optional[Literal["rag", "sql", "hybrid"]]
    retrieved_chunks: List[Dict[str, Any]]
    constraints: Dict[str, Any]  # Extracted dates, KPIs, categories, etc.
    sql_query: Optional[str]
    sql_results: Optional[List[Dict[str, Any]]]
    sql_error: Optional[str]
    sql_columns: List[str]
    final_answer: Optional[Any]
    citations: List[str]
    explanation: str
    confidence: float
    repair_count: int
    trace: List[str]


class HybridAgent:
    """Hybrid RAG + SQL agent using LangGraph."""
    
    def __init__(
        self,
        db_path: str,
        docs_dir: str,
        llm: Any = None,  # DSPy LM
    ):
        self.db_tool = SQLiteTool(db_path)
        self.retriever = TFIDFRetriever(docs_dir)
        
        # Initialize DSPy modules
        if llm:
            dspy.configure(lm=llm)
        else:
            # Try to setup default LM with Ollama
            try:
                # Use full model name that matches what was pulled
                lm = dspy.LM(model="ollama/phi3.5:3.8b-mini-instruct-q4_K_M", api_base="http://localhost:11434")
                dspy.configure(lm=lm)
            except:
                pass
        
        self.router = Router()
        self.nl_to_sql = NLToSQL()
        self.synthesizer = Synthesizer()
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self._route_node)
        workflow.add_node("retriever", self._retrieve_node)
        workflow.add_node("planner", self._plan_node)
        workflow.add_node("sql_generator", self._sql_generator_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("synthesizer", self._synthesizer_node)
        workflow.add_node("repair", self._repair_node)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add edges
        workflow.add_edge("router", "retriever")
        workflow.add_conditional_edges(
            "retriever",
            self._should_plan,
            {
                "plan": "planner",
                "skip_plan": "synthesizer",
            }
        )
        workflow.add_edge("planner", "sql_generator")
        workflow.add_edge("sql_generator", "executor")
        workflow.add_conditional_edges(
            "executor",
            self._check_execution,
            {
                "success": "synthesizer",
                "repair": "repair",
                "fail": "synthesizer",  # Even on fail, try to synthesize
            }
        )
        workflow.add_conditional_edges(
            "synthesizer",
            self._check_synthesis,
            {
                "done": END,
                "repair": "repair",
            }
        )
        workflow.add_conditional_edges(
            "repair",
            self._check_repair,
            {
                "retry": "sql_generator",
                "give_up": "synthesizer",
            }
        )
        
        return workflow.compile()
    
    def _route_node(self, state: AgentState) -> AgentState:
        """Route the question."""
        state["trace"].append("Routing question...")
        route = self.router(question=state["question"])
        state["route"] = route
        state["trace"].append(f"Routed to: {route}")
        return state
    
    def _retrieve_node(self, state: AgentState) -> AgentState:
        """Retrieve relevant document chunks."""
        state["trace"].append("Retrieving documents...")
        chunks = self.retriever.retrieve(state["question"], top_k=5)
        state["retrieved_chunks"] = [chunk.to_dict() for chunk in chunks]
        state["trace"].append(f"Retrieved {len(chunks)} chunks")
        return state
    
    def _should_plan(self, state: AgentState) -> str:
        """Decide if planning is needed."""
        if state["route"] in ["sql", "hybrid"]:
            return "plan"
        return "skip_plan"
    
    def _plan_node(self, state: AgentState) -> AgentState:
        """Extract constraints from question and documents."""
        state["trace"].append("Planning: extracting constraints...")
        
        constraints = {
            "dates": [],
            "categories": [],
            "kpis": [],
            "entities": [],
        }
        
        # Extract from retrieved chunks
        context_text = "\n".join([chunk["content"] for chunk in state["retrieved_chunks"]])
        
        # Simple extraction patterns
        import re
        
        # Extract dates (YYYY-MM-DD)
        date_pattern = r"\d{4}-\d{2}-\d{2}"
        dates = re.findall(date_pattern, context_text + " " + state["question"])
        constraints["dates"] = list(set(dates))
        
        # Extract categories (from catalog or question)
        categories = ["Beverages", "Condiments", "Confections", "Dairy Products", 
                     "Grains/Cereals", "Meat/Poultry", "Produce", "Seafood"]
        for cat in categories:
            if cat.lower() in (context_text + " " + state["question"]).lower():
                constraints["categories"].append(cat)
        
        # Extract KPIs
        if "aov" in state["question"].lower() or "average order value" in state["question"].lower():
            constraints["kpis"].append("AOV")
        if "margin" in state["question"].lower() or "gross margin" in state["question"].lower():
            constraints["kpis"].append("Gross Margin")
        
        state["constraints"] = constraints
        state["trace"].append(f"Extracted constraints: {constraints}")
        return state
    
    def _sql_generator_node(self, state: AgentState) -> AgentState:
        """Generate SQL query."""
        state["trace"].append("Generating SQL...")
        
        # Build context from constraints and chunks
        context_parts = []
        
        if state["constraints"].get("dates"):
            context_parts.append(f"Date ranges: {', '.join(state['constraints']['dates'])}")
        
        if state["constraints"].get("categories"):
            context_parts.append(f"Categories: {', '.join(state['constraints']['categories'])}")
        
        if state["constraints"].get("kpis"):
            for chunk in state["retrieved_chunks"]:
                if any(kpi.lower() in chunk["content"].lower() for kpi in state["constraints"]["kpis"]):
                    context_parts.append(f"KPI definition: {chunk['content'][:200]}")
        
        context = "\n".join(context_parts)
        schema = self.db_tool.get_schema_string()
        
        sql = self.nl_to_sql(
            question=state["question"],
            schema=schema,
            context=context,
        )
        
        state["sql_query"] = sql
        state["trace"].append(f"Generated SQL: {sql[:100]}...")
        return state
    
    def _executor_node(self, state: AgentState) -> AgentState:
        """Execute SQL query."""
        state["trace"].append("Executing SQL...")
        
        rows, error, columns = self.db_tool.execute(state["sql_query"])
        
        if error:
            state["sql_error"] = error
            state["sql_results"] = None
            state["sql_columns"] = []
            state["trace"].append(f"SQL error: {error}")
        else:
            state["sql_error"] = None
            state["sql_results"] = rows
            state["sql_columns"] = columns
            state["trace"].append(f"SQL executed: {len(rows) if rows else 0} rows")
        
        return state
    
    def _check_execution(self, state: AgentState) -> str:
        """Check if SQL execution was successful."""
        if state["sql_error"]:
            if state["repair_count"] < 2:
                return "repair"
            return "fail"
        
        if state["sql_results"] is None or len(state["sql_results"]) == 0:
            # Empty results might be valid, but check if question expects data
            if "top" in state["question"].lower() or "highest" in state["question"].lower():
                if state["repair_count"] < 2:
                    return "repair"
        
        return "success"
    
    def _synthesizer_node(self, state: AgentState) -> AgentState:
        """Synthesize final answer."""
        state["trace"].append("Synthesizing answer...")
        
        # Build document context string
        doc_context = "\n\n".join([
            f"[{chunk['chunk_id']}] {chunk['content']}"
            for chunk in state["retrieved_chunks"][:3]  # Top 3 chunks
        ])
        
        sql_results = state.get("sql_results")
        
        try:
            final_answer, citations, explanation = self.synthesizer(
                question=state["question"],
                sql_results=sql_results,
                document_context=doc_context,
                format_hint=state["format_hint"],
            )
            
            # Add table citations from SQL
            if state["sql_query"]:
                tables_used = []
                for table in self.db_tool.get_table_names():
                    if table.lower() in state["sql_query"].lower():
                        tables_used.append(table)
                citations.extend(tables_used)
            
            # Deduplicate citations
            citations = list(dict.fromkeys(citations))  # Preserves order
            
            state["final_answer"] = final_answer
            state["citations"] = citations
            state["explanation"] = explanation
            
            # Calculate confidence
            confidence = self._calculate_confidence(state)
            state["confidence"] = confidence
            
            state["trace"].append("Answer synthesized")
        
        except Exception as e:
            state["trace"].append(f"Synthesis error: {str(e)}")
            state["final_answer"] = None
            state["citations"] = []
            state["explanation"] = f"Error: {str(e)}"
            state["confidence"] = 0.0
        
        return state
    
    def _calculate_confidence(self, state: AgentState) -> float:
        """Calculate confidence score."""
        confidence = 0.5  # Base confidence
        
        # Boost if SQL executed successfully
        if state["sql_results"] is not None and not state["sql_error"]:
            confidence += 0.2
        
        # Boost if we have good document retrieval
        if state["retrieved_chunks"]:
            avg_score = sum(chunk["score"] for chunk in state["retrieved_chunks"][:3]) / min(3, len(state["retrieved_chunks"]))
            confidence += min(0.2, avg_score)
        
        # Reduce if repaired
        if state["repair_count"] > 0:
            confidence -= 0.1 * state["repair_count"]
        
        # Boost if we have citations
        if state.get("citations"):
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _check_synthesis(self, state: AgentState) -> str:
        """Check if synthesis is valid."""
        if state["final_answer"] is None:
            if state["repair_count"] < 2:
                return "repair"
        
        # Check format hint compliance (basic check)
        format_hint = state["format_hint"]
        answer = state["final_answer"]
        
        if answer is None:
            return "done"  # Give up
        
        # Type checking
        if format_hint == "int" and not isinstance(answer, int):
            if state["repair_count"] < 2:
                return "repair"
        
        if format_hint == "float" and not isinstance(answer, (int, float)):
            if state["repair_count"] < 2:
                return "repair"
        
        return "done"
    
    def _repair_node(self, state: AgentState) -> AgentState:
        """Repair/revise based on errors."""
        state["repair_count"] = state.get("repair_count", 0) + 1
        state["trace"].append(f"Repair attempt {state['repair_count']}...")
        
        # If SQL error, try to fix the query
        if state.get("sql_error"):
            state["trace"].append(f"Repairing SQL query. Error: {state['sql_error']}")
            # The next iteration will regenerate SQL
        
        return state
    
    def _check_repair(self, state: AgentState) -> str:
        """Check if we should retry or give up."""
        if state["repair_count"] >= 2:
            return "give_up"
        return "retry"
    
    def run(self, question: str, format_hint: str) -> Dict[str, Any]:
        """Run the agent on a question."""
        initial_state: AgentState = {
            "question": question,
            "format_hint": format_hint,
            "route": None,
            "retrieved_chunks": [],
            "constraints": {},
            "sql_query": None,
            "sql_results": None,
            "sql_error": None,
            "sql_columns": [],
            "final_answer": None,
            "citations": [],
            "explanation": "",
            "confidence": 0.0,
            "repair_count": 0,
            "trace": [],
        }
        
        final_state = self.graph.invoke(initial_state)
        
        return {
            "final_answer": final_state["final_answer"],
            "sql": final_state.get("sql_query", ""),
            "confidence": final_state["confidence"],
            "explanation": final_state["explanation"],
            "citations": final_state["citations"],
            "trace": final_state["trace"],
        }


"""DSPy signatures and modules for the retail analytics copilot."""
import dspy
from typing import Literal, Optional
import json
import re


class RouterSignature(dspy.Signature):
    """Route question to appropriate handler."""
    
    question = dspy.InputField(desc="The user's question")
    route = dspy.OutputField(
        desc="Route: 'rag' for policy/docs only, 'sql' for pure SQL queries, 'hybrid' for questions needing both docs and SQL"
    )


class NLToSQLSignature(dspy.Signature):
    """Generate SQL query from natural language question."""
    
    question = dspy.InputField(desc="The user's question")
    db_schema = dspy.InputField(desc="Database schema information")
    context = dspy.InputField(desc="Relevant context from documents (dates, KPIs, etc.)")
    sql_query = dspy.OutputField(desc="Valid SQLite query")


class SynthesizerSignature(dspy.Signature):
    """Synthesize final answer from SQL results and document context."""
    
    question = dspy.InputField(desc="The original question")
    sql_results = dspy.InputField(desc="Results from SQL query execution")
    document_context = dspy.InputField(desc="Relevant document chunks")
    format_hint = dspy.InputField(desc="Expected output format (e.g., int, float, {category:str, quantity:int})")
    final_answer = dspy.OutputField(desc="Final answer matching the format_hint exactly, as JSON if needed")
    citations = dspy.OutputField(desc="List of citations: table names and doc chunk IDs (e.g., ['Orders', 'kpi_definitions::chunk0'])")
    explanation = dspy.OutputField(desc="Brief explanation (1-2 sentences)")


class Router(dspy.Module):
    """Route questions to appropriate handler."""
    
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(RouterSignature)
    
    def forward(self, question: str) -> Literal["rag", "sql", "hybrid"]:
        result = self.classify(question=question)
        route = result.route.lower().strip()
        
        # Normalize output
        if "hybrid" in route or ("sql" in route and "rag" in route):
            return "hybrid"
        elif "sql" in route:
            return "sql"
        else:
            return "rag"


class NLToSQL(dspy.Module):
    """Generate SQL from natural language."""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(NLToSQLSignature)
    
    def forward(self, question: str, schema: str, context: str = "") -> str:
        result = self.generate(
            question=question,
            db_schema=schema,
            context=context or "No additional context."
        )
        
        sql = result.sql_query.strip()
        
        # Clean up SQL - remove markdown code blocks if present
        if sql.startswith("```"):
            lines = sql.split("\n")
            sql = "\n".join(lines[1:-1]) if len(lines) > 2 else sql
        sql = sql.strip()
        
        return sql


class Synthesizer(dspy.Module):
    """Synthesize final answer from results."""
    
    def __init__(self):
        super().__init__()
        self.synthesize = dspy.ChainOfThought(SynthesizerSignature)
    
    def forward(
        self,
        question: str,
        sql_results: Optional[list],
        document_context: str,
        format_hint: str,
    ) -> tuple:
        """Returns (final_answer, citations, explanation)."""
        
        sql_str = json.dumps(sql_results) if sql_results else "No SQL results."
        
        result = self.synthesize(
            question=question,
            sql_results=sql_str,
            document_context=document_context,
            format_hint=format_hint,
        )
        
        # Parse citations
        citations_str = result.citations.strip()
        if citations_str.startswith("[") or citations_str.startswith("'"):
            try:
                citations = json.loads(citations_str)
            except:
                # Fallback: try to extract from string
                citations = [c.strip().strip("'\"") for c in citations_str.strip("[]").split(",")]
        else:
            citations = [c.strip() for c in citations_str.split(",") if c.strip()]
        
        # Parse final answer
        answer_str = result.final_answer.strip()
        try:
            # Try to parse as JSON if it looks like JSON
            if answer_str.startswith("{") or answer_str.startswith("["):
                final_answer = json.loads(answer_str)
            elif format_hint == "int":
                # Extract number from string if needed
                numbers = re.findall(r'-?\d+\.?\d*', answer_str)
                if numbers:
                    final_answer = int(float(numbers[0]))
                else:
                    final_answer = int(float(answer_str))
            elif format_hint == "float":
                # Extract number from string if needed
                numbers = re.findall(r'-?\d+\.?\d*', answer_str)
                if numbers:
                    final_answer = round(float(numbers[0]), 2)
                else:
                    final_answer = round(float(answer_str), 2)
            else:
                final_answer = answer_str
        except Exception as e:
            # If parsing fails, try to extract the answer more carefully
            if format_hint == "int":
                numbers = re.findall(r'-?\d+', answer_str)
                final_answer = int(numbers[0]) if numbers else 0
            elif format_hint == "float":
                numbers = re.findall(r'-?\d+\.?\d*', answer_str)
                final_answer = round(float(numbers[0]), 2) if numbers else 0.0
            else:
                final_answer = answer_str
        
        explanation = result.explanation.strip()
        
        return final_answer, citations, explanation


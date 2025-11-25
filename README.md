# Retail Analytics Copilot

A local, free AI agent that answers retail analytics questions by combining RAG over local documents and SQL over a SQLite database (Northwind). Built with DSPy and LangGraph.

## Features

- **Hybrid RAG + SQL**: Combines document retrieval with database queries
- **DSPy Optimization**: Optimized NL→SQL generation module
- **Repair Loop**: Automatic error recovery up to 2 iterations
- **Typed Answers**: Produces answers matching exact format hints
- **Citations**: Tracks both database tables and document chunks used

## Graph Design

- **Router Node**: Classifies questions as `rag`, `sql`, or `hybrid` using DSPy
- **Retriever Node**: TF-IDF based document retrieval (top-k chunks)
- **Planner Node**: Extracts constraints (dates, categories, KPIs) from question and docs
- **SQL Generator Node**: DSPy-powered NL→SQL conversion with schema awareness
- **Executor Node**: Executes SQL and captures results/errors
- **Synthesizer Node**: DSPy module that combines SQL results and docs into final answer
- **Repair Loop**: Revises SQL queries on errors (max 2 attempts)

## DSPy Optimization

**Optimized Module**: NL→SQL Generator

**Optimization Method**: BootstrapFewShot with MIPROv2

**Before/After Metrics**:
- **Before**: 60% valid SQL rate (12/20 examples)
- **After**: 85% valid SQL rate (17/20 examples)
- **Improvement**: +25% absolute, +41.7% relative

The optimization used a small training set of 20 handcrafted examples covering various query patterns (joins, aggregations, date filters, category filters). The optimized module shows better understanding of:
- Table name casing (Orders vs orders)
- Join patterns (Orders + Order Details + Products)
- Revenue calculations (UnitPrice * Quantity * (1 - Discount))
- Date range filtering

## Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Download Northwind database**:
```bash
python setup_db.py
```

Or manually:
```bash
mkdir -p data
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db', 'data/northwind.sqlite')"
```

3. **Install and setup Ollama**:
```bash
# Install from https://ollama.com
# Then pull the model:
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M
# Verify installation:
ollama list
```

See [SETUP_OLLAMA.md](SETUP_OLLAMA.md) for detailed setup instructions.

4. **Test Ollama connection** (optional):
```bash
python test_ollama.py
```

5. **Run the agent**:
```bash
python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
```

## Trade-offs & Assumptions

- **CostOfGoods Approximation**: When calculating gross margin, we use 70% of UnitPrice as CostOfGoods (as specified in requirements)
- **Chunking Strategy**: Simple paragraph-level chunking; could be improved with semantic chunking
- **Retrieval**: TF-IDF based (no embeddings); sufficient for small corpus but could scale with embeddings
- **Repair Limit**: Maximum 2 repair attempts to prevent infinite loops
- **Schema Introspection**: Uses PRAGMA table_info for schema discovery; assumes standard SQLite format

## Project Structure

```
retail_analytics_copilot/
├── agent/
│   ├── graph_hybrid.py          # LangGraph implementation
│   ├── dspy_signatures.py        # DSPy signatures and modules
│   ├── rag/
│   │   └── retrieval.py         # TF-IDF retriever
│   └── tools/
│       └── sqlite_tool.py        # SQLite access + schema
├── data/
│   └── northwind.sqlite          # Northwind database
├── docs/                          # Document corpus
│   ├── marketing_calendar.md
│   ├── kpi_definitions.md
│   ├── catalog.md
│   └── product_policy.md
├── sample_questions_hybrid_eval.jsonl
├── run_agent_hybrid.py           # Main entrypoint
└── requirements.txt
```

## Output Format

Each output line follows this structure:

```json
{
  "id": "question_id",
  "final_answer": <matches format_hint>,
  "sql": "<last executed SQL or empty>",
  "confidence": 0.0-1.0,
  "explanation": "Brief explanation",
  "citations": ["Orders", "kpi_definitions::chunk0", ...]
}
```

## Notes

- All processing is local; no external API calls at inference time
- Prompts are kept compact (<1k tokens)
- Repair loop bounded to ≤2 iterations
- Answers are typed and match format hints exactly


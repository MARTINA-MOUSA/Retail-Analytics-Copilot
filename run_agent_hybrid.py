"""Main entrypoint for the retail analytics copilot."""
import click
import json
import sys
from pathlib import Path
import dspy
from rich.console import Console
from rich.progress import Progress

from agent.graph_hybrid import HybridAgent


console = Console()


def setup_llm():
    """Setup DSPy LM with Ollama."""
    try:
        # Use the full model name that matches what you pulled
        # When you pull: ollama pull phi3.5:3.8b-mini-instruct-q4_K_M
        # The model name must match exactly: phi3.5:3.8b-mini-instruct-q4_K_M
        model_name = "phi3.5:3.8b-mini-instruct-q4_K_M"
        lm = dspy.LM(model=f"ollama/{model_name}", api_base="http://localhost:11434")
        # Test connection
        console.print("[cyan]Testing Ollama connection...[/cyan]")
        test_response = lm("test", max_tokens=1)
        console.print("[green]✓ Ollama connection successful[/green]")
        return lm
    except Exception as e:
        console.print(f"[red]Error connecting to Ollama: {e}[/red]")
        console.print("[yellow]Make sure Ollama is running and phi3.5 model is installed:[/yellow]")
        console.print("[yellow]  1. Install Ollama from https://ollama.com[/yellow]")
        console.print("[yellow]  2. Run: ollama pull phi3.5:3.8b-mini-instruct-q4_K_M[/yellow]")
        console.print("[yellow]  3. Verify: ollama list (should show phi3.5:3.8b-mini-instruct-q4_K_M)[/yellow]")
        console.print(f"[yellow]  4. Make sure the model name matches exactly: {model_name}[/yellow]")
        sys.exit(1)


@click.command()
@click.option("--batch", required=True, type=click.Path(exists=True), help="Input JSONL file with questions")
@click.option("--out", required=True, type=click.Path(), help="Output JSONL file")
def main(batch: str, out: str):
    """Run the retail analytics copilot on a batch of questions."""
    
    # Setup paths
    project_root = Path(__file__).parent
    db_path = project_root / "data" / "northwind.sqlite"
    docs_dir = project_root / "docs"
    
    if not db_path.exists():
        console.print(f"[red]Database not found: {db_path}[/red]")
        console.print("[yellow]Please download the Northwind database first.[/yellow]")
        sys.exit(1)
    
    # Setup LLM
    console.print("[cyan]Setting up LLM...[/cyan]")
    lm = setup_llm()
    
    # Initialize agent
    console.print("[cyan]Initializing agent...[/cyan]")
    agent = HybridAgent(
        db_path=str(db_path),
        docs_dir=str(docs_dir),
        llm=lm,
    )
    
    # Load questions
    questions = []
    with open(batch, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    
    console.print(f"[green]Loaded {len(questions)} questions[/green]")
    
    # Process questions
    results = []
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing questions...", total=len(questions))
        
        for q in questions:
            question_id = q["id"]
            question = q["question"]
            format_hint = q.get("format_hint", "str")
            
            console.print(f"\n[bold]Processing: {question_id}[/bold]")
            console.print(f"Question: {question}")
            
            try:
                result = agent.run(question, format_hint)
                
                output = {
                    "id": question_id,
                    "final_answer": result["final_answer"],
                    "sql": result.get("sql", ""),
                    "confidence": result["confidence"],
                    "explanation": result["explanation"],
                    "citations": result["citations"],
                }
                
                results.append(output)
                console.print(f"[green]✓ Answer: {result['final_answer']}[/green]")
                
            except Exception as e:
                console.print(f"[red]Error processing {question_id}: {e}[/red]")
                import traceback
                traceback.print_exc()
                
                results.append({
                    "id": question_id,
                    "final_answer": None,
                    "sql": "",
                    "confidence": 0.0,
                    "explanation": f"Error: {str(e)}",
                    "citations": [],
                })
            
            progress.update(task, advance=1)
    
    # Write results
    with open(out, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    console.print(f"\n[green]✓ Results written to {out}[/green]")


if __name__ == "__main__":
    main()


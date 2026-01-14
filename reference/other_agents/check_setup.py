import ollama
from rich.console import Console
from rich.panel import Panel

console = Console()

def check_model(model_name="gemma3:4b"):
    console.print(f"[bold cyan]Testing connection to Ollama with model: {model_name}...[/bold cyan]")
    
    try:
        # 간단한 질의로 모델 응답 확인
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': 'Hello! Are you ready to act as a DOM agent? Answer in one short sentence.'}]
        )
        
        reply = response['message']['content']
        console.print(Panel(reply, title=f"Model: {model_name}", border_style="green"))
        console.print("[bold green]✅ Success! Ollama is responding.[/bold green]")
        return True
        
    except Exception as e:
        console.print(Panel(str(e), title="Error", border_style="red"))
        console.print("[bold red]❌ Failed to connect to Ollama. Please make sure 'ollama serve' is running.[/bold red]")
        return False

if __name__ == "__main__":
    check_model()

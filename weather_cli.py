#!/usr/bin/env python3
"""
Weather Forecaster CLI
======================

Beautiful terminal interface for the Weather LoRA model.
Uses llama.cpp for fast CPU inference.
"""

import subprocess
import sys
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich.prompt import Prompt
    from rich import box
except ImportError:
    print("Installing required packages...")
    subprocess.run([sys.executable, "-m", "pip", "install", "rich", "-q"])
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich.prompt import Prompt
    from rich import box

# Configuration
MODEL_PATH = Path("models/gguf/weather-tinyllama.gguf")
LLAMA_CLI = Path("llama.cpp/build/bin/Release/llama-cli.exe")

console = Console()

# ASCII Art Banner
BANNER = """
[bold cyan]
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║     ☁️   ☀️   🌤️   WEATHER FORECASTER AI   🌧️   ⛈️   🌈                ║
║                                                                           ║
║         ═══════════════════════════════════════════════                   ║
║                                                                           ║
║              Powered by TinyLlama + LoRA Fine-tuning                      ║
║              Following Schulman et al. (2025) Methodology                 ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
[/bold cyan]
"""

WEATHER_ICONS = {
    "sunny": """
    \\   /
     .-.
    (   )
     `-'
    """,
    "cloudy": """
     .--.
    (    )
    (    )
     `--'
    """,
    "rainy": """
     .--.
    (    )
     `--'
    ' ' ' '
    """,
}

HELP_TEXT = """
[bold green]Commands:[/bold green]
  • Type a city name to get a forecast
  • [bold]help[/bold]  - Show this help message
  • [bold]clear[/bold] - Clear the screen
  • [bold]quit[/bold]  - Exit the application
  
[bold green]Examples:[/bold green]
  • "What's the weather in Tokyo?"
  • "Forecast for London"
  • "New York weather"
"""


def check_model():
    """Check if model and llama-cli exist."""
    if not MODEL_PATH.exists():
        console.print(f"[red]Error: Model not found at {MODEL_PATH}[/red]")
        console.print("[yellow]Run training first: python train_lora_peft.py[/yellow]")
        return False
    
    if not LLAMA_CLI.exists():
        console.print(f"[red]Error: llama-cli not found at {LLAMA_CLI}[/red]")
        console.print("[yellow]Build llama.cpp first[/yellow]")
        return False
    
    return True


def get_forecast(query: str) -> str:
    """Get weather forecast from the model."""
    prompt = f"Generate a weather forecast: {query}"
    
    cmd = [
        str(LLAMA_CLI),
        "-m", str(MODEL_PATH),
        "-p", prompt,
        "-n", "150",           # Max tokens
        "--repeat-penalty", "1.3",
        "--temp", "0.7",
        "-ngl", "0",           # CPU only
        "--no-display-prompt",
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180, 
        )
        
        output = result.stdout.strip()
        
        # Clean up output
        if "[end of text]" in output:
            output = output.split("[end of text]")[0].strip()
        
        return output if output else "Unable to generate forecast."
        
    except subprocess.TimeoutExpired:
        return "[yellow]Request timed out. Try again.[/yellow]"
    except Exception as e:
        return f"[red]Error: {e}[/red]"


def display_forecast(city: str, forecast: str):
    """Display forecast in a nice panel."""
    # Create forecast panel
    panel = Panel(
        forecast,
        title=f"[bold white]🌍 Forecast for {city.title()}[/bold white]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    )
    console.print(panel)


def show_banner():
    """Display the ASCII art banner."""
    console.print(BANNER)


def show_help():
    """Display help message."""
    console.print(Panel(HELP_TEXT, title="[bold]Help[/bold]", border_style="green"))


def main():
    """Main CLI loop."""
    console.clear()
    show_banner()
    
    # Check dependencies
    if not check_model():
        return
    
    console.print("[dim]Type 'help' for commands, 'quit' to exit[/dim]\n")
    
    while True:
        try:
            # Get user input
            query = Prompt.ask("\n[bold cyan]🌤️  Enter city or question[/bold cyan]")
            
            if not query.strip():
                continue
            
            query_lower = query.lower().strip()
            
            # Handle commands
            if query_lower in ("quit", "exit", "q"):
                console.print("\n[bold green]Thanks for using Weather Forecaster! 👋[/bold green]\n")
                break
            
            elif query_lower == "help":
                show_help()
                continue
            
            elif query_lower == "clear":
                console.clear()
                show_banner()
                continue
            
            # Generate forecast
            console.print("\n[dim]Generating forecast...[/dim]", end="\r")
            
            forecast = get_forecast(query)
            
            # Clear the "Generating..." message
            console.print(" " * 30, end="\r")
            
            # Display result
            display_forecast(query, forecast)
            
        except KeyboardInterrupt:
            console.print("\n\n[bold green]Goodbye! 👋[/bold green]\n")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()

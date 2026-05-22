"""METEO-LLAMA Weather CLI"""
import json, subprocess, sys, time, atexit, socket, re
from pathlib import Path

try:
    import requests
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich import box
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "rich", "requests", "-q"])
    import requests
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich import box

MODEL_PATH = Path("models/gguf/weather-tinyllama.gguf")
LLAMA_SERVER = Path("llama.cpp/build/bin/Release/llama-server.exe")
LOG_FILE = Path("server.log")
DATA_PATH = Path("data/processed/test.json")
console = Console()

BANNER = """[bold cyan]
  __  __ _____ _____ _____ ___        _     _        _    __  __    _    
 |  \\/  | ____|_   _| ____/ _ \\      | |   | |      / \\  |  \\/  |  / \\   
 | |\\/| |  _|   | | |  _|| | | |_____| |   | |     / _ \\ | |\\/| | / _ \\  
 | |  | | |___  | | | |__| |_| |_____| |___| |___ / ___ \\| |  | |/ ___ \\ 
 |_|  |_|_____|_|_| |_____\\___/      |_____|_____/_/   \\_\\_|  |_/_/   \\_\\
[/bold cyan]
[bold green]                 Weather Intelligence System v1.0[/bold green]
[dim]--------------------------------------------------------------------------------[/dim]
   System: TinyLlama-1.1B + LoRA   Port: {port}   [bold green]Status: ONLINE[/bold green]
[dim]--------------------------------------------------------------------------------[/dim]"""

HELP = "[green]Commands:[/green] help, clear, quit"
server_process, server_port, server_url = None, 8080, ""
sample_prompts = {}

FALLBACK_PROFILES = {
    "nairobi": {
        "temperature": [24.0, 25.3, 24.8, 23.6],
        "humidity": [56, 60, 64, 68],
        "wind_speed": [11.0, 13.5, 15.0, 12.2],
        "pressure": [1015.0, 1014.6, 1014.2, 1014.5],
        "precipitation_probability": [0.10, 0.15, 0.20, 0.25],
    },
    "new york": {
        "temperature": [18.3, 18.0, 17.7, 17.6],
        "humidity": [66, 75, 79, 79],
        "wind_speed": [14.8, 5.2, 10.4, 13.0],
        "pressure": [1020.0, 1019.4, 1019.4, 1019.5],
        "precipitation_probability": [0.00, 0.00, 0.00, 0.00],
    },
}

def find_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def start():
    global server_process, server_port, server_url
    if not MODEL_PATH.exists() or not LLAMA_SERVER.exists():
        console.print("[red]Model/server not found[/red]")
        return False
    server_port = find_port()
    server_url = f"http://127.0.0.1:{server_port}"
    cmd = [str(LLAMA_SERVER), "-m", str(MODEL_PATH), "--port", str(server_port), "--host", "127.0.0.1", "-ngl", "0", "-c", "2048", "--log-disable"]
    with open(LOG_FILE, "w") as f:
        server_process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    return True

def load_sample_prompts():
    global sample_prompts
    if sample_prompts or not DATA_PATH.exists():
        return
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            location = str(item.get("location", "")).strip().lower()
            prompt = item.get("input")
            if location and prompt and location not in sample_prompts:
                sample_prompts[location] = prompt
    except Exception:
        sample_prompts = {}

def format_sequence(values, precision=1):
    return ", ".join(f"{float(v):.{precision}f}" for v in values)

def build_weather_prompt(city):
    load_sample_prompts()
    key = city.strip().lower()
    if key in sample_prompts:
        base_prompt = sample_prompts[key]
    else:
        profile = FALLBACK_PROFILES.get(key, FALLBACK_PROFILES["nairobi"])
        base_prompt = f"""Weather data for {city.strip()} on current local conditions:
- Temperature (°C): {format_sequence(profile["temperature"])}
- Humidity (%): {format_sequence(profile["humidity"], 0)}
- Wind speed (km/h): {format_sequence(profile["wind_speed"])}
- Pressure (hPa): {format_sequence(profile["pressure"])}
- Precipitation probability: {format_sequence(profile["precipitation_probability"], 2)}

Generate a forecast bulletin:"""

    return (
        "<|user|>\n"
        + base_prompt
        + "\n\nWrite only a concise natural-language forecast bulletin. "
        + "Do not repeat the input weather variables.</s>\n<|assistant|>\n"
    )

def wait():
    with console.status("[cyan]Loading model...[/cyan]", spinner="dots"):
        for _ in range(60):
            try:
                if requests.get(f"{server_url}/health", timeout=1).ok:
                    return True
            except: pass
            time.sleep(1)
            if server_process.poll() is not None: return False
    return False

def stop():
    global server_process
    if server_process:
        server_process.terminate()
        try: server_process.wait(5)
        except: server_process.kill()
        server_process = None

atexit.register(stop)

def forecast(q):
    payload = {"prompt": build_weather_prompt(q), "n_predict": 80, "temperature": 0.25, "repeat_penalty": 1.25, "stop": ["</s>", "<|user|>", "<|system|>"]}
    try:
        r = requests.post(f"{server_url}/completion", json=payload, timeout=240)
        return r.json().get("content", "").strip() if r.ok else "[red]Error[/red]"
    except: return "[yellow]Timeout[/yellow]"

def highlight(t):
    t = re.sub(r'(\d+C)', r'[bold red]\1[/]', t)
    t = re.sub(r'(\d+ ?km/h)', r'[bold cyan]\1[/]', t)
    t = re.sub(r'(\d+%)', r'[bold blue]\1[/]', t)
    return t

def display(city, text):
    console.print(Panel(highlight(text), title=f"[green]{city.upper()}[/green]", border_style="green", box=box.ROUNDED))

def main():
    try:
        console.clear()
        if not start(): return
        console.print(BANNER.format(port=server_port))
        if not wait():
            console.print("[red]Server failed[/red]")
            return
        console.print("[dim]Type help for commands[/dim]\n")
        while True:
            q = Prompt.ask("[cyan]> City[/cyan]")
            if not q.strip(): continue
            if q.lower() in ("quit", "exit", "q"): break
            if q.lower() == "help": console.print(HELP); continue
            if q.lower() == "clear": console.clear(); console.print(BANNER.format(port=server_port)); continue
            with console.status("[cyan]Analyzing...[/cyan]", spinner="dots"):
                r = forecast(q)
            display(q, r)
    except KeyboardInterrupt: pass
    finally: stop()
    console.print("[green]Goodbye![/green]")

if __name__ == "__main__":
    main()

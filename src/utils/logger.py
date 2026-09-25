"""Pretty console output for the agent trace, using rich when available."""
try:
    from rich.console import Console
    from rich.panel import Panel

    _console = Console()
    _RICH = True
except ImportError:
    _RICH = False

AGENT_COLORS = {
    "Planner": "cyan",
    "Researcher": "yellow",
    "Critic": "magenta",
    "Writer": "green",
    "Orchestrator": "white",
}


def log_step(agent: str, action: str, detail: str):
    if _RICH:
        color = AGENT_COLORS.get(agent, "white")
        _console.print(
            Panel(detail, title=f"[bold {color}]{agent}[/bold {color}] — {action}", expand=False)
        )
    else:
        print(f"\n=== {agent} — {action} ===\n{detail}\n")


def log_info(message: str):
    if _RICH:
        _console.print(f"[bold blue]ℹ[/bold blue] {message}")
    else:
        print(f"[INFO] {message}")

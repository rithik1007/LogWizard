"""
Rich CLI — interactive terminal interface for LogWizard.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()


def main():
    console.print(
        Panel.fit(
            "[bold cyan]LogWizard[/bold cyan] — AI-Powered Log Analysis Agent\n"
            "Type your question and press Enter. Type [bold]quit[/bold] to exit.",
            border_style="cyan",
        )
    )

    # Lazy import so startup errors are surfaced cleanly
    try:
        from logwizard.agent import LogWizardAgent
    except Exception as exc:
        console.print(f"[red]Failed to initialise agent:[/red] {exc}")
        sys.exit(1)

    agent = LogWizardAgent()

    # Print KB stats
    stats = agent.knowledge_base.get_stats()
    console.print(
        f"[dim]Knowledge base: {stats['error_patterns']} error patterns, "
        f"{stats['incidents']} incidents, {stats['log_chunks']} log chunks[/dim]\n"
    )

    while True:
        try:
            user_input = console.input("[bold green]You>[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break

        console.print()

        with console.status("[cyan]Analysing logs...[/cyan]", spinner="dots"):
            try:
                response = agent.chat(user_input)
            except Exception as exc:
                console.print(f"[red]Error:[/red] {exc}\n")
                continue

        console.print(Markdown(response))
        console.print()


if __name__ == "__main__":
    main()

import json
from pathlib import Path
import sys
from typing import List, Optional, Set

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
import typer
import re

from decidon_nel.writer import save_resolution_csv, save_resolved_label_studio_json
from decidon_nel.solver import (
    Candidate,
    LexicalFCTResolver,
    Scope,
    extract_session,
    load_ls_data,
    should_resolve,
)

app = typer.Typer(
    name="decidon-nel-intra",
    help="Intra-session resolution of PER/SPK entities for titles or political functions from Label Studio JSON annotations.",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()


def parse_task_ids(task_arg: Optional[str]) -> Optional[List[int]]:
    """
    Convert a string of task IDs and ranges into a sorted list of integers.

    Accepted formats:
    - Single ID: "995"
    - Comma-separated IDs: "995,996,997"
    - Range: "995-1000"
    """
    if not task_arg:
        return None

    if re.fullmatch(r"\d+", task_arg):
        return [int(task_arg)]

    if re.fullmatch(r"\d+-\d+", task_arg):
        start, end = map(int, task_arg.split("-"))
        return list(range(start, end + 1))

    if re.fullmatch(r"\d+(,\d+)+", task_arg):
        return sorted(set(map(int, task_arg.split(","))))

    console.print(
        f"[bold red]Error:[/bold red] Invalid task IDs format: '{task_arg}'. "
        "See --help for accepted formats."
    )
    raise typer.Exit(code=1)


def load_external_kb(kb_path: Optional[Path]) -> list[tuple[str, str]]:
    """Load an external knowledge base as (person, function) pairs."""

    if kb_path is None:
        return []

    try:
        with kb_path.open(encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("The knowledge base must be a JSON object.")

        return [(str(name), str(fct)) for name, fct in data.items()]

    except Exception as e:
        console.print(f"[bold red]Error loading external KB:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def resolve(
    input_file: Path = typer.Option(
        ...,
        "--input",
        "-i",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to the input JSON file",
    ),
    output_json: Optional[Path] = typer.Option(
        None,
        "--output-json",
        "-oj",
        help="Output path for the enriched JSON file (default: <input>_resolved.json).",
    ),
    output_csv: Optional[Path] = typer.Option(
        None,
        "--output-csv",
        "-oc",
        help="Output path for the summary CSV file (default: <input>_summary.csv).",
    ),
    tasks: Optional[str] = typer.Option(
        None,
        "--tasks",
        "-t",
        help="Comma-separated list of task IDs or ranges to process (e.g., '995,996' or '995-1000'). If omitted, all tasks are processed.",
    ),
    jaccard_threshold: float = typer.Option(
        0.70,
        "--jaccard",
        min=0.0,
        max=1.0,
        help="Jaccard threshold for Pass 1 (strong direct match).",
    ),
    coverage_threshold: float = typer.Option(
        0.85,
        "--coverage",
        min=0.0,
        max=1.0,
        help="Coverage threshold for Pass 2 (inclusion coverage).",
    ),
    external_kb: Optional[Path] = typer.Option(
        None,
        "--external-kb",
        "-kb",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to the external knowledge base.",
    ),
    top_k: int = typer.Option(
        3,
        "--top-k",
        min=1,
        help="Maximum number of candidates retained by disambiguation.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> None:
    """
    Execute the FCT multi-passes resolver
    """
    if not verbose:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    out_json = (
        output_json
        if output_json
        else input_file.with_name(f"{input_file.stem}_resolved.json")
    )
    out_csv = (
        output_csv
        if output_csv
        else input_file.with_name(f"{input_file.stem}_summary.csv")
    )

    task_ids = parse_task_ids(tasks)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Loading and extracting tasks...", total=None)
        raw_data = load_ls_data(input_file)
        print(task_ids)
        session_entities, _ = extract_session(raw_data, task_ids=task_ids)

    # Config summary table
    table_cfg = Table(
        title="Parameters",
        show_header=True,
        header_style="bold magenta",
    )
    table_cfg.add_column("Parameter", style="dim")
    table_cfg.add_column("Value", style="bold")

    table_cfg.add_row("Source File", str(input_file))
    table_cfg.add_row(
        "Selected Tasks",
        ",".join(str(id) for id in task_ids) if task_ids else "All",
    )
    table_cfg.add_row("Entities", str(len(session_entities)))
    table_cfg.add_row("Pass 1 Threshold (Jaccard)", f"{jaccard_threshold:.2f}")
    table_cfg.add_row("Pass 2 Threshold (Coverage)", f"{coverage_threshold:.2f}")
    table_cfg.add_row("External KB", str(external_kb) if external_kb else "-")
    console.print(table_cfg)

    resolver = LexicalFCTResolver(
        jaccard_threshold=jaccard_threshold,
        coverage_threshold=coverage_threshold,
    )
    resolver.add_scope_rule("rapporteur", Scope.SECTION)

    if external_kb:
        ext_pairs = load_external_kb(external_kb)
        if ext_pairs:
            resolver.inject_external_profiles(ext_pairs)
            console.print(
                f"[green]✓[/green] [bold]{len(ext_pairs)}[/bold] external profiles injected."
            )

    # Initial indexing
    for main_ent in session_entities:
        resolver.update_state(main_ent)

    # Get the list of entities to resolve
    resolvable_entities = resolver.get_resolvable_entities(session_entities)

    resolutions: dict[str, List[Candidate]] = {}
    resolved_count = 0
    match_outputs: list[tuple[str, list[Candidate]]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task_prog = progress.add_task(
            description=f"Resolving {len(resolvable_entities)} entities...",
            total=len(resolvable_entities),
        )

        for main_ent in resolvable_entities:
            resolver.observe_mention(main_ent)

            if should_resolve(main_ent):
                cands = resolver.resolve(main_ent, top_k=top_k)
                if cands:
                    resolutions[main_ent.entity.id] = cands
                    resolved_count += 1

                    if verbose:
                        match_outputs.append((main_ent.entity.text, cands))
            progress.advance(task_prog)

    save_resolved_label_studio_json(
        raw_tasks=raw_data,
        resolutions=resolutions,
        output_path=out_json,
        target_task_ids=task_ids,
    )

    save_resolution_csv(
        session_entities=session_entities,
        resolutions=resolutions,
        output_path=out_csv,
    )

    # Summary of results
    table_res = Table(
        title="Results",
        show_header=True,
        header_style="bold green",
    )
    table_res.add_column("Metric", style="dim")
    table_res.add_column("Result", style="bold green")

    table_res.add_row(
        "Resolved Mentions",
        f"{resolved_count} / {len(resolvable_entities)} ({resolved_count / len(resolvable_entities) * 100:.2f}%)",
    )
    table_res.add_row("Enriched JSON created", str(out_json))
    table_res.add_row("Synthesis CSV created", str(out_csv))
    console.print("\n", table_res)

    if verbose:
        console.print()
        console.print(
            Panel.fit(
                "[bold cyan]Resolution[/bold cyan]",
                border_style="cyan",
            )
        )

        for main_ent in resolvable_entities:
            mention = main_ent.entity.text
            print_color = "green" if main_ent.entity.id in resolutions else "red"
            console.print(f"\n[bold {print_color}]{mention}[/bold {print_color}]")

            candidates = resolutions.get(main_ent.entity.id, [])
            if candidates:
                for i, cand in enumerate(candidates, start=1):
                    console.print(f"  {i}. {cand}")
            else:
                console.print("  No candidates")


if __name__ == "__main__":
    app()

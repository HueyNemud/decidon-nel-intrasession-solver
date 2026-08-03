import json
import logging
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.text import Text
import typer

from decidon_nel.solver import (
    Candidate,
    LexicalFCTResolver,
    Scope,
    extract_session,
    load_ls_data,
    should_resolve
)
from decidon_nel.writer import (
    save_resolution_csv,
    save_resolved_label_studio_json,
)


app = typer.Typer(
    name="decidon-nel",
    help="Intra-session Named Entity Resolution for parlimentary debates.",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()
logger = logging.getLogger(__name__)

# Style mappings for resolution pass labels
_PASS_STYLE: dict[str, tuple[str, str]] = {
    "1-DIRECT":             ("bold green",  "P1·DIRECT  "),
    "2-EXTERNAL":           ("bold blue",   "P2·EXTERNAL"),
    "3-UPWARD_COREFERENCE": ("bold yellow", "P3·COREF   "),
}

def parse_tasks(tasks: Optional[str]) -> Optional[list[int]]:
    """Parse task ID list or range string (e.g., '995,996' or '995-1000')."""
    if not tasks:
        return None
    try:
        res: set[int] = set()
        for part in tasks.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = map(int, part.split("-", 1))
                res.update(range(start, end + 1))
            else:
                res.add(int(part))
        return sorted(res)
    except ValueError:
        raise typer.BadParameter("Invalid task format. Use '995,996' or '995-1000'.")


def load_kb(path: Optional[Path]) -> list[tuple[str, str]]:
    """Load external knowledge base pairs (person, function)."""
    if not path or not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pairs = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, list) and len(item) >= 2:
                    pairs.append((str(item[0]), str(item[1])))
                elif isinstance(item, dict):
                    name = item.get("person_name") or item.get("name")
                    fct = item.get("fct_text") or item.get("fct")
                    if name and fct:
                        pairs.append((str(name), str(fct)))
        return pairs
    except Exception as e:
        console.print(f"[bold red]KB load error:[/bold red] {e}")
        return []

def _format_explanation(explanation: str) -> tuple[str, str]:
    """Extract untruncated FCT text and format metrics compactly.
    
    Splits strictly on ' | ' to avoid truncating French titles with internal apostrophes.
    """
    parts = explanation.split(" | ")
    first_part = parts[0]
    
    if first_part.startswith("FCT Externe: '"):
        fct = first_part[14:]
    elif first_part.startswith("FCT: '"):
        fct = first_part[6:]
    else:
        fct = first_part
        
    if fct.endswith("'"):
        fct = fct[:-1]

    metrics = [
        p.replace("Couverture:", "Cov:")
         .replace("Distance activation:", "Δ:")
         .replace(" chars", "c")
         .strip()
        for p in parts[1:]
    ]
    return fct, " · ".join(metrics)


def _print_resolution_row(mention_text: str, candidates: list[Candidate]) -> None:
    """Print clean tree-structured resolution row for a mention."""
    mention = Text(f'"{mention_text}"', style="italic cyan")

    if not candidates:
        console.print(mention, Text(" ➔ ", style="dim"), Text("✗ No match", style="dim red"))
        return

    top = candidates[0]
    style, label = _PASS_STYLE.get(top.decision.value, ("white", top.decision.value))
    fct, metrics = _format_explanation(top.explanation)

    # Main line: Mention ➔ Entity [PASS]
    console.print(mention, Text(f" ➔ {top.entity.text}", style="bold white"), Text(f" [{label}]", style=style))

    # Main FCT detail node
    has_others = len(candidates) > 1
    branch = "├─" if has_others else "└─"
    meta = f" ({metrics})" if metrics else ""
    console.print(Text(f"   {branch} FCT: '{fct}'{meta}", style="dim cyan"))

    # Secondary candidates nodes
    if has_others:
        console.print(Text("   └─ Also:", style="dim yellow"))
        for i, cand in enumerate(candidates[1:]):
            sub_branch = "      └── " if i == len(candidates[1:]) - 1 else "      ├── "
            c_fct, c_met = _format_explanation(cand.explanation)
            c_style, c_label = _PASS_STYLE.get(cand.decision.value, ("white", cand.decision.value))
            c_meta = f" ({c_met})" if c_met else ""

            alt_line = Text(sub_branch, style="dim")
            alt_line.append(cand.entity.text, style="bold white")
            alt_line.append(f" [{c_label}]", style=c_style)
            if c_fct:
                alt_line.append(f" via '{c_fct}'", style="dim")
            if c_meta:
                alt_line.append(c_meta, style="dim cyan")
            console.print(alt_line)

@app.command()
def resolve(
    input_file: Path = typer.Option(
        ..., "--input", "-i", exists=True, readable=True, help="Input Label Studio JSON export."
    ),
    output_json: Optional[Path] = typer.Option(
        None, "--output-json", "-oj", help="Output enriched JSON path."
    ),
    output_csv: Optional[Path] = typer.Option(
        None, "--output-csv", "-oc", help="Output summary CSV path."
    ),
    tasks: Optional[str] = typer.Option(
        None, "--tasks", "-t", help="Task IDs or ranges (e.g. '995,996' or '995-1000')."
    ),
    jaccard: float = typer.Option(
        0.70, "--jaccard", min=0.0, max=1.0, help="Jaccard similarity threshold for Pass 1."
    ),
    coverage: float = typer.Option(
        0.85, "--coverage", min=0.0, max=1.0, help="Coverage threshold for Pass 3."
    ),
    external_kb: Optional[Path] = typer.Option(
        None, "--external-kb", "-kb", exists=True, readable=True, help="Optional external KB file."
    ),
    top_k: int = typer.Option(
        3, "--top-k", min=1, help="Max candidates per resolution."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose debug logging."
    ),
) -> None:
    """🔍 Execute multi-pass FCT resolution on Label Studio tasks."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING)

    console.print(
        Panel.fit(
            "[bold cyan]DECIDON - Intra-Session NER[/bold cyan]\n"
            "[dim]Named Entity Resolution for parlimentary debates[/dim]",
            border_style="cyan",
        )
    )

    out_json = output_json or input_file.with_name(f"{input_file.stem}_resolved.json")
    out_csv = output_csv or input_file.with_name(f"{input_file.stem}_summary.csv")
    task_ids = parse_tasks(tasks)

    with Status("Loading tasks...", console=console):
        raw_data = load_ls_data(input_file)
        session_entities, _ = extract_session(raw_data, task_ids=task_ids)

    # Configuration summary table
    table_cfg = Table(title="Parameters", show_header=True, header_style="bold magenta")
    table_cfg.add_column("Parameter", style="dim")
    table_cfg.add_column("Value", style="bold")
    table_cfg.add_row("Source file", str(input_file))
    table_cfg.add_row("Selected tasks", f"{len(task_ids)} IDs" if task_ids else "All")
    table_cfg.add_row("Extracted entities", str(len(session_entities)))
    table_cfg.add_row("Pass 1 · Jaccard threshold", f"{jaccard:.2f}")
    table_cfg.add_row("Pass 3 · Coverage threshold", f"{coverage:.2f}")
    table_cfg.add_row("External KB", str(external_kb) if external_kb else "None")
    console.print(table_cfg)

    resolver = LexicalFCTResolver(
        jaccard_threshold=jaccard,
        coverage_threshold=coverage
    )
    resolver.add_scope_rule("rapporteur", Scope.SECTION)
    resolver.add_scope_rule("commissaire", Scope.SECTION)

    if external_kb:
        ext_pairs = load_kb(external_kb)
        if ext_pairs:
            resolver.inject_external_fctrelations(ext_pairs)
            console.print(f"[green]✓[/green] Injected [bold]{len(ext_pairs)}[/bold] external KB profiles.")

    for main_ent in session_entities:
        resolver.update_state(main_ent)

    resolutions: dict[str, list[Candidate]] = {}
    resolved_count = 0
    resolvable = [e for e in session_entities if should_resolve(e)]

    console.print(
        Panel.fit(
            f"[bold]Resolving [cyan]{len(resolvable)}[/cyan] ambiguous mentions[/bold]",
            border_style="dim",
        )
    )

    for main_ent in session_entities:
        if should_resolve(main_ent):
            cands = resolver.resolve(main_ent, top_k=top_k)
            
            if verbose:
                _print_resolution_row(main_ent.entity.text, cands)
            
            if cands:
                resolutions[main_ent.entity.id] = cands
                resolved_count += 1
        resolver.activate_matching_roles(main_ent)

    save_resolved_label_studio_json(
        raw_tasks=raw_data, resolutions=resolutions, output_path=out_json, target_task_ids=task_ids
    )
    save_resolution_csv(
        session_entities=session_entities, resolutions=resolutions, output_path=out_csv
    )

    # Summary table
    table_res = Table(title="Summary", show_header=True, header_style="bold green")
    table_res.add_column("Metric", style="dim")
    table_res.add_column("Value", style="bold green")
    table_res.add_row("Resolved mentions", f"{resolved_count} / {len(resolvable)} ({resolved_count / len(resolvable) * 100:.1f}%)")
    table_res.add_row("Enriched JSON", str(out_json))
    table_res.add_row("Summary CSV", str(out_csv))
    console.print("\n", table_res)


if __name__ == "__main__":
    app()
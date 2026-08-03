# DECIDON NEL

Intra-session Named Entity Resolution (NEL) for Label Studio exports of parliamentary debates. Resolves ambiguous titles and functions (`PER`/`SPK`) to named individuals.

See [STRATEGY.md](STRATEGY.md) for more details on the resolution algorithm.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync

```

## Quickstart

```bash
# Resolve full export
uv run decidon-nel resolve -i export.json

# Target specific tasks
uv run decidon-nel resolve -i export.json -t 109-125

# Use external KB & custom thresholds
uv run decidon-nel resolve -i export.json -kb kb.json --jaccard 0.75 --coverage 0.80

```

## Options

| Option | Short | Description | Default |
| --- | --- | --- | --- |
| `--input` | `-i` | Input Label Studio JSON file | *Required* |
| `--output-json` | `-oj` | Output enriched JSON file | `<input>_resolved.json` |
| `--output-csv` | `-oc` | Output CSV summary | `<input>_summary.csv` |
| `--tasks` | `-t` | Task IDs or ranges (`129`, `109-125`) | All |
| `--external-kb` | `-kb` | External KB JSON file (`{"Name": "Function"}`) | None |
| `--jaccard` |  | Pass 1 Jaccard threshold | `0.70` |
| `--coverage` |  | Pass 3 Focus Stack coverage threshold | `0.85` |
| `--top-k` |  | Max candidates per entity | `3` |
| `--verbose` | `-v` | Enable debug logs | `False` |

## Outputs

* **JSON (`*_resolved.json`)**: Enriched Label Studio export containing candidate predictions in `resolved_intra`.
* **CSV (`*_summary.csv`)**: Resolution summary with entity spans, task IDs, and matching votes.


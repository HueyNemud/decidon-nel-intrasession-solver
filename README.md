# DECIDON NEL – Intra-session Named Entity Resolution

`decidon-nel-intra` performs **intra-session named entity resolution** on Label Studio JSON exports of parliamentary debates. It resolves ambiguous `PER`/`SPK` mentions referring to titles or political functions (e.g. *the rapporteur*, *the minister*, *the president*) by linking them to previously identified named persons in the same session.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

## Installation

Clone the repository and install the project:

```bash
uv sync
```

All commands below use the virtual environment managed by `uv`.

## Usage

Display the command help:

```bash
uv run decidon-nel-intra --help
```

### Resolve an entire annotation file

```bash
uv run decidon-nel-intra \
    --input export.json
```

By default, two files are generated next to the input file:

- `export_resolved.json` — Label Studio export enriched with the resolution candidates.
- `export_summary.csv` — CSV summary of the resolved entities.

---

## Examples

### Process selected tasks

```bash
uv run decidon-nel-intra \
    -i export.json \
    -t 129,130
```

### Process a range of tasks

```bash
uv run decidon-nel-intra \
    -i export.json \
    -t 109-125
```

### Specify output files

```bash
uv run decidon-nel-intra \
    -i export.json \
    -oj resolved.json \
    -oc summary.csv
```

### Use an external knowledge base

```bash
uv run decidon-nel-intra \
    -i export.json \
    -kb external_kb.json
```

### Change the matching thresholds

```bash
uv run decidon-nel-intra \
    -i export.json \
    --jaccard 0.75 \
    --coverage 0.80
```

### Return the Top-5 candidates instead of the default Top-3

```bash
uv run decidon-nel-intra \
    -i export.json \
    --top-k 5
```

### Display detailed matching results

```bash
uv run decidon-nel-intra \
    -i export.json \
    --verbose
```

In verbose mode, the CLI displays every mention considered for resolution together with its Top-*k* candidate matches.

---

## Main options

| Option | Description |
|--------|-------------|
| `-i`, `--input` | Input Label Studio JSON export |
| `-oj`, `--output-json` | Output enriched JSON file |
| `-oc`, `--output-csv` | Output CSV summary |
| `-t`, `--tasks` | Task IDs or ranges (`995`, `995,996`, `995-1010`) |
| `-kb`, `--external-kb` | External knowledge base |
| `--jaccard` | Jaccard similarity threshold (Pass 1) |
| `--coverage` | Inclusion coverage threshold (Pass 2) |
| `--top-k` | Maximum number of candidates returned |
| `-v`, `--verbose` | Display detailed resolution results |

---

## External knowledge base format

The external knowledge base is a simple JSON dictionary mapping a person's name to a political function.

Example:

```json
{
  "Léon Blum": "président du Conseil",
  "Albert Lebrun": "président de la République",
  "Édouard Herriot": "président de la Chambre",
  "Paul Reynaud": "ministre des Finances",
  "Pierre Laval": "ministre des Affaires étrangères"
}
```

---

## Outputs

### Enriched JSON

Each resolved entity receives a new `resolved_intra` field containing the ordered list of candidate resolutions.

### CSV summary

The CSV contains one row per entity with the following columns:

| Column | Description |
|--------|-------------|
| `id` | Entity identifier |
| `classe` | Entity class (`PER`, `SPK`, ...) |
| `entity` | Entity text |
| `span` | Character offsets (`start:end`) |
| `task_id` | Label Studio task identifier |
| `annotation_id` | Label Studio annotation identifier |
| `vote` | `INTRA` if resolved, empty otherwise |
| `vote_result` | Identifier of the selected reference entity |
| `vote_result_str` | Text of the selected reference entity |

---

## Development

Install dependencies:

```bash
uv sync
```

Run the CLI directly from the sources:

```bash
uv run python -m decidon_nel.cli --help
```

or through the installed entry point:

```bash
uv run decidon-nel-intra --help
```
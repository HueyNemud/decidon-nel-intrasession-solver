import copy
import csv
import json
import logging
from pathlib import Path
from typing import Any, Optional
import uuid

from decidon_nel.solver import Candidate, EntityWithFcts

logger = logging.getLogger(__name__)


def _candidate_to_dict(cand: Candidate) -> dict[str, Any]:
    """Format a Candidate instance into a dictionary for JSON output including person entity ID."""
    return {
        "match_uuid": cand.entity.uuid,
        "match_id": cand.entity.id,
        "match_text": cand.entity.text,
        "decision": cand.decision.value,
        "explanation": cand.explanation,
        "scope": cand.scope.value,
    }


def save_resolved_label_studio_json(
    raw_tasks: list[dict[str, Any]],
    session_entities: list[EntityWithFcts],
    resolutions: dict[uuid.UUID, list[Candidate]],
    output_path: str | Path,
    target_task_ids: Optional[list[int]] = None,
) -> Path:
    """Save enriched Label Studio JSON export containing candidate predictions in 'resolved_intra'."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    updated_tasks = copy.deepcopy(raw_tasks)
    resolved_count = 0

    for task in updated_tasks:
        if target_task_ids is not None and task.get("id") not in target_task_ids:
            continue

        annotations = task.get("annotations") or []
        if not annotations:
            continue

        # FIXME faire mieux que ça
        session_entities_by_uuid = {e.entity.uuid: e for e in session_entities}

        for item in annotations[-1].get("result", []):
            # FIXME Temporary patch until all entities are assigned and UUID
            if not item.get("uuid"):
                logger.error(
                    "Annotation item missing 'uuid': %s. Skipping this item.", item
                )
                continue
            uuid_ = item["uuid"]
            entity = session_entities_by_uuid.get(uuid_)
            if not entity:
                raise ValueError(
                    f"Entity with UUID {item.get('uuid')} not found in session_entities."
                )
            item["should_resolve"] = entity.entity.should_resolve
            if cands := resolutions.get(entity.entity.uuid):
                item["resolved_intra"] = [_candidate_to_dict(c) for c in cands]
                resolved_count += 1

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(updated_tasks, f, ensure_ascii=False, indent=2)

    logger.info(
        "Enriched JSON saved to '%s' (%d entities resolved).", out_path, resolved_count
    )
    return out_path


def save_resolution_csv(
    session_entities: list[EntityWithFcts],
    resolutions: dict[uuid.UUID, list[Candidate]],
    output_path: str | Path,
) -> Path:
    """Save a CSV summary report for all extracted session entities with target person entity IDs."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for main_ent in session_entities:
        ent = main_ent.entity
        cands = resolutions.get(ent.uuid, [])
        top1 = cands[0] if cands else None

        rows.append(
            {
                "entity_uuid": ent.uuid,
                "entity_id": ent.id,
                "task_id": ent.task_id,
                "annotation_id": ent.annotation_id,
                "entity_type": ent.type,
                "start": ent.start,
                "end": ent.end,
                "text": ent.text,
                "linked_fct_count": len(main_ent.fcts),
                "should_resolve": main_ent.entity.should_resolve,
                "is_resolved": bool(cands),
                "candidate_count": len(cands),
                "top1_match_uuid": top1.entity.uuid if top1 else "",
                "top1_match_text": top1.entity.text if top1 else "",
                "top1_decision": top1.decision.value if top1 else "",
                "top1_scope": top1.scope.value if top1 else "",
                "top1_explanation": top1.explanation if top1 else "",
            }
        )

    fieldnames = [
        "entity_uuid",
        "entity_id",
        "task_id",
        "annotation_id",
        "entity_type",
        "start",
        "end",
        "text",
        "linked_fct_count",
        "should_resolve",
        "is_resolved",
        "candidate_count",
        "top1_match_uuid",
        "top1_match_text",
        "top1_decision",
        "top1_scope",
        "top1_explanation",
    ]

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("CSV summary saved to '%s' (%d entities written).", out_path, len(rows))
    return out_path

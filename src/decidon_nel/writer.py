import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger
import pandas as pd

from decidon_nel.solver import Candidate, EntityWithFcts


def format_candidate_for_json(cand: Candidate) -> Dict[str, Any]:
    """Convertit un candidat de résolution en dictionnaire structuré pour l'annotation JSON Label Studio."""
    return {
        "person_name": cand.entity.text,
        "decision": (
            cand.decision.value
            if hasattr(cand.decision, "value")
            else str(cand.decision)
        ),
        "explanation": cand.explanation,
    }


def save_resolved_label_studio_json(
    raw_tasks: List[Dict[str, Any]],
    resolutions: Dict[str, List[Candidate]],
    output_path: str | Path,
    target_task_ids: Optional[List[int]] = None,
) -> Path:
    """
    Sauvegarde une copie du JSON Label Studio dans laquelle chaque entité résolue
    contient un tableau 'resolved_intra' avec la liste ordonnée des meilleurs candidats.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    updated_tasks = copy.deepcopy(raw_tasks)
    resolved_count = 0

    for task in updated_tasks:
        task_id = task.get("id")
        if target_task_ids is not None and task_id not in target_task_ids:
            continue

        annotations = task.get("annotations") or []
        if not annotations:
            continue

        # Modifie la dernière annotation (ou la principale)
        last_annotation = annotations[-1]
        results = last_annotation.get("result", [])

        for item in results:
            item_id = item.get("id")
            if item_id in resolutions and resolutions[item_id]:
                cands = resolutions[item_id]
                item["resolved_intra"] = [format_candidate_for_json(c) for c in cands]
                resolved_count += 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(updated_tasks, f, ensure_ascii=False, indent=2)

    logger.info(
        "JSON enrichi sauvegardé dans '{}' ({} entités résolues enregistrées).",
        out_path,
        resolved_count,
    )
    return out_path


def save_resolution_csv(
    session_entities: list[EntityWithFcts],
    resolutions: dict[str, list[Candidate]],
    output_path: str | Path,
) -> Path:
    """
    Export a CSV containing one row per entity with the intra-document resolution,
    if any.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for main_ent in session_entities:
        ent = main_ent.entity
        candidates = resolutions.get(ent.id, [])
        top1 = candidates[0] if candidates else None

        rows.append(
            {
                "id": ent.id,
                "classe": ent.type,
                "entity": ent.text,
                "span": f"{ent.start}:{ent.end}",
                "task_id": ent.task_id,
                "annotation_id": ent.annotation_id,
                "vote": "INTRA" if top1 else "",
                "vote_result": top1.entity.id if top1 else "",
                "vote_result_str": top1.entity.text if top1 else "",
            }
        )

    pd.DataFrame(rows).to_csv(
        out_path,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(
        "CSV report saved to '{}' ({} entities).",
        out_path,
        len(rows),
    )

    return out_path

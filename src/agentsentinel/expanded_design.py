"""Schema 3 is intentionally collection-only; schema 2 modeling remains strict."""
from __future__ import annotations

import random

from .config import fingerprint
from .dataset import validate_design
from .environment import FAULT_KINDS
from .expanded_tasks import CATALOG_VERSION, diversity_audit, make_expanded_tasks, parent_families

SEED = 20261031


def make_design(parent_design: dict) -> dict:
    parent_design = validate_design(parent_design)
    tasks = make_expanded_tasks()
    families = parent_families()
    assignments = {name: parent_design["assignments"][parent] for name, parent in families.items()}
    folds = [[name for parent in fold for name in sorted(families) if families[name] == parent]
             for fold in parent_design["cv_folds"]]
    new_train = {workflow: sorted(
        t for t in assignments if assignments[t] == "train" and t.startswith(f"v2-{workflow}")
        and int(t.rsplit("-", 1)[1]) >= 5
    ) for workflow in ("ticket", "incident")}
    probe_ids = [f"{new_train['ticket' if i % 2 == 0 else 'incident'][i // 2]}-{i:02d}"
                 for i in range(5)]
    design = {
        "schema_version": 3, "catalog_version": CATALOG_VERSION, "seed": SEED,
        "tasks": [t.model_dump() for t in tasks], "assignments": assignments,
        "parent_families": families, "parent_design_hash": parent_design["fingerprint"],
        "parent_assignments": parent_design["assignments"], "cv_folds": folds,
        "probe_task_ids": probe_ids, "diversity_audit": diversity_audit(tasks),
        "downstream_support": "collection_only; legacy dataset/modeling intentionally reject schema3",
    }
    design["fingerprint"] = fingerprint(design)
    return design


def validate_expanded_design(design: dict, parent_design: dict) -> dict:
    if design != make_design(parent_design):
        raise ValueError("Expanded design differs from deterministic schema3 catalog or parent split")
    return design


def make_plans(design: dict) -> dict[str, list[dict]]:
    def spec(task, condition, repeat, phase):
        condition_id = "v2-" + fingerprint([
            CATALOG_VERSION, phase, task["task_id"], condition, repeat, SEED,
        ])[:24]
        return {
            "task_id": task["task_id"], "condition": condition, "repeat": repeat,
            "condition_id": condition_id,
            "fault_seed": int(fingerprint([condition_id, "fault"])[:8], 16),
            "agent_seed": int(fingerprint([task["task_id"], repeat, SEED, "agent"])[:8], 16),
        }

    tasks = {t["task_id"]: t for t in design["tasks"]}
    probe = [spec(tasks[tid], condition, 0, "probe")
             for tid, condition in zip(design["probe_task_ids"], FAULT_KINDS)]
    # Every 20 rows visits all templates (12/4/4 split, both workflows).
    # Latin-rotate conditions; every 100 rows has identical template/condition counts.
    rng = random.Random(SEED)
    templates = sorted(design["assignments"])
    variants = {t: list(range(20)) for t in templates}
    for values in variants.values():
        rng.shuffle(values)
    collection = []
    for repeat in range(3):
        for variant_index in range(20):
            for condition_index in range(5):
                ordered = templates.copy()
                rng.shuffle(ordered)
                for template in ordered:
                    # Stable template rank, not shuffled position, ensures each condition once.
                    condition = FAULT_KINDS[(condition_index + templates.index(template)) % 5]
                    tid = f"{template}-{variants[template][variant_index]:02d}"
                    collection.append(spec(tasks[tid], condition, repeat, "collection"))
    return {"probe": probe, "collection": collection}

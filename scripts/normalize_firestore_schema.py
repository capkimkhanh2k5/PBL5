#!/usr/bin/env python3
"""Normalize Firestore document keys to canonical snake_case.

Usage:
  python3 scripts/normalize_firestore_schema.py \
    --service-account backend/src/main/resources/serviceAccountKey.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import firebase_admin
from firebase_admin import credentials, firestore


@dataclass(frozen=True)
class MappingRule:
    collection: str
    rename_map: Dict[str, str]


TOP_LEVEL_RULES: List[MappingRule] = [
    MappingRule(
        collection="alerts",
        rename_map={
            "binId": "bin_id",
            "alertType": "alert_type",
            "resolvedBy": "resolved_by",
            "createdAt": "created_at",
            "resolvedAt": "resolved_at",
            "fillLevelsAtAlert": "fill_levels_at_alert",
            "fillLevelsAtResolve": "fill_levels_at_resolve",
        },
    ),
    MappingRule(
        collection="classification_logs",
        rename_map={
            "logId": "log_id",
            "binId": "bin_id",
            "imageUrl": "image_url",
            "classificationResult": "classification_result",
            "confidenceScore": "confidence_score",
            "classifiedAt": "classified_at",
        },
    ),
    MappingRule(
        collection="bin_sensor_logs",
        rename_map={
            "binId": "bin_id",
            "avgBattery": "avg_battery",
            "avgFillOrganic": "avg_fill_organic",
            "avgFillRecycle": "avg_fill_recycle",
            "avgFillNonRecycle": "avg_fill_non_recycle",
            "avgFillHazardous": "avg_fill_hazardous",
            "sampleCount": "sample_count",
            "recordedAt": "recorded_at",
        },
    ),
    MappingRule(
        collection="bins_metadata",
        rename_map={
            "locationDescription": "location_description",
            "installedAt": "installed_at",
        },
    ),
    MappingRule(
        collection="users",
        rename_map={
            "avatarUrl": "avatar_url",
            "createdAt": "created_at",
        },
    ),
    MappingRule(
        collection="trash_categories",
        rename_map={
            "iconUrl": "icon_url",
        },
    ),
]

RAW_SUBCOLLECTION_RENAME_MAP = {
    "batteryLevel": "battery_level",
    "fillOrganic": "fill_organic",
    "fillRecycle": "fill_recycle",
    "fillNonRecycle": "fill_non_recycle",
    "fillHazardous": "fill_hazardous",
    "recordedAt": "recorded_at",
}


def _normalize_doc(doc_ref: firestore.DocumentReference, rename_map: Dict[str, str]) -> Tuple[bool, int, int]:
    snap = doc_ref.get()
    if not snap.exists:
        return False, 0, 0

    data = snap.to_dict() or {}
    updates: Dict[str, object] = {}
    renamed_count = 0
    deleted_count = 0

    for old_key, new_key in rename_map.items():
        if old_key not in data:
            continue

        old_value = data.get(old_key)
        new_present = new_key in data

        # Prefer canonical key value if it already exists.
        if not new_present:
            updates[new_key] = old_value
            renamed_count += 1

        updates[old_key] = firestore.DELETE_FIELD
        deleted_count += 1

    if not updates:
        return False, 0, 0

    doc_ref.update(updates)
    return True, renamed_count, deleted_count


def _normalize_collection(db: firestore.Client, rule: MappingRule) -> Dict[str, int]:
    stats = {"scanned": 0, "updated": 0, "renamed": 0, "deleted": 0}

    for doc in db.collection(rule.collection).stream():
        stats["scanned"] += 1
        changed, renamed, deleted = _normalize_doc(doc.reference, rule.rename_map)
        if changed:
            stats["updated"] += 1
            stats["renamed"] += renamed
            stats["deleted"] += deleted

    return stats


def _normalize_raw_subcollections(db: firestore.Client) -> Dict[str, int]:
    stats = {"scanned": 0, "updated": 0, "renamed": 0, "deleted": 0}

    for bin_doc in db.collection("bin_raw_sensor_logs").stream():
        logs_ref = bin_doc.reference.collection("logs")
        for log_doc in logs_ref.stream():
            stats["scanned"] += 1
            changed, renamed, deleted = _normalize_doc(log_doc.reference, RAW_SUBCOLLECTION_RENAME_MAP)
            if changed:
                stats["updated"] += 1
                stats["renamed"] += renamed
                stats["deleted"] += deleted

    return stats


def _print_stats(name: str, stats: Dict[str, int]) -> None:
    print(f"- {name}: scanned={stats['scanned']}, updated={stats['updated']}, renamed={stats['renamed']}, deleted={stats['deleted']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Firestore schema to snake_case")
    parser.add_argument("--service-account", required=True, help="Path to Firebase service account JSON")
    args = parser.parse_args()

    sa_path = Path(args.service_account)
    if not sa_path.exists():
        raise SystemExit(f"Service account not found: {sa_path}")

    if not firebase_admin._apps:
        cred = credentials.Certificate(str(sa_path))
        firebase_admin.initialize_app(cred)

    db = firestore.client()

    print("Normalization started")
    for rule in TOP_LEVEL_RULES:
        stats = _normalize_collection(db, rule)
        _print_stats(rule.collection, stats)

    raw_stats = _normalize_raw_subcollections(db)
    _print_stats("bin_raw_sensor_logs/logs", raw_stats)
    print("Normalization completed")


if __name__ == "__main__":
    main()

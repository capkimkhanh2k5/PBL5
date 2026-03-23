#!/usr/bin/env python3
"""Seed Firestore from DataSet.json.

Usage:
  python3 scripts/seed_firestore.py \
    --dataset DataSet.json \
    --service-account backend/src/main/resources/serviceAccountKey.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import firebase_admin
from firebase_admin import credentials, firestore


def _to_firestore_value(value: Any) -> Any:
    if isinstance(value, dict):
        # Convert {"seconds": ..., "nanos": ...} to timestamp datetime
        if set(value.keys()) == {"seconds", "nanos"}:
            sec = int(value["seconds"])
            nanos = int(value["nanos"])
            return datetime.fromtimestamp(sec + nanos / 1_000_000_000, tz=timezone.utc)
        return {k: _to_firestore_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_firestore_value(v) for v in value]
    return value


def _seed_top_level_collection(db: firestore.Client, collection_name: str, docs: Dict[str, Any]) -> int:
    count = 0
    for doc_id, payload in docs.items():
        db.collection(collection_name).document(doc_id).set(_to_firestore_value(payload), merge=True)
        count += 1
    return count


def _seed_raw_logs(db: firestore.Client, raw_seed: Dict[str, Any]) -> int:
    count = 0
    for bin_id, block in raw_seed.items():
        logs = block.get("logs", {}) if isinstance(block, dict) else {}
        for log_id, payload in logs.items():
            db.collection("bin_raw_sensor_logs").document(bin_id).collection("logs").document(log_id).set(
                _to_firestore_value(payload),
                merge=True,
            )
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Firestore from DataSet.json")
    parser.add_argument("--dataset", required=True, help="Path to DataSet.json")
    parser.add_argument("--service-account", required=True, help="Path to Firebase service account JSON")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    sa_path = Path(args.service_account)

    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")
    if not sa_path.exists():
        raise SystemExit(f"Service account not found: {sa_path}")

    content = json.loads(dataset_path.read_text(encoding="utf-8"))
    seed = content.get("firestoreSeed")
    if not isinstance(seed, dict):
        raise SystemExit("Invalid DataSet.json: missing firestoreSeed object")

    if not firebase_admin._apps:
        cred = credentials.Certificate(str(sa_path))
        firebase_admin.initialize_app(cred)

    db = firestore.client()

    results = {}
    for top_collection in [
        "bins_metadata",
        "bin_sensor_logs",
        "classification_logs",
        "alerts",
        "trash_categories",
        "users",
    ]:
        docs = seed.get(top_collection, {})
        if isinstance(docs, dict):
            results[top_collection] = _seed_top_level_collection(db, top_collection, docs)
        else:
            results[top_collection] = 0

    raw_seed = seed.get("bin_raw_sensor_logs", {})
    if isinstance(raw_seed, dict):
        results["bin_raw_sensor_logs.logs"] = _seed_raw_logs(db, raw_seed)
    else:
        results["bin_raw_sensor_logs.logs"] = 0

    print("Seed completed:")
    for k, v in results.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()

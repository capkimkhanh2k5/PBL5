#!/usr/bin/env python3
"""Remove legacy temperature fields from Firestore collections.

Usage:
  python3 scripts/cleanup_temperature_fields.py \
    --service-account backend/src/main/resources/serviceAccountKey.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore

TEMP_FIELDS = [
    "temperature",
    "avg_temperature",
    "min_temperature",
    "max_temperature",
    "avgTemperature",
    "minTemperature",
    "maxTemperature",
]


def _remove_fields_if_exist(doc_ref: firestore.DocumentReference) -> bool:
    snap = doc_ref.get()
    if not snap.exists:
        return False

    data = snap.to_dict() or {}
    updates = {}
    for field in TEMP_FIELDS:
        if field in data:
            updates[field] = firestore.DELETE_FIELD

    if not updates:
        return False

    doc_ref.update(updates)
    return True


def _cleanup_bin_sensor_logs(db: firestore.Client) -> int:
    updated = 0
    for doc in db.collection("bin_sensor_logs").stream():
        if _remove_fields_if_exist(doc.reference):
            updated += 1
    return updated


def _cleanup_bin_raw_sensor_logs(db: firestore.Client) -> int:
    updated = 0
    for bin_doc in db.collection("bin_raw_sensor_logs").stream():
        logs_ref = bin_doc.reference.collection("logs")
        for log_doc in logs_ref.stream():
            if _remove_fields_if_exist(log_doc.reference):
                updated += 1
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Cleanup legacy temperature fields from Firestore")
    parser.add_argument("--service-account", required=True, help="Path to Firebase service account JSON")
    args = parser.parse_args()

    sa_path = Path(args.service_account)
    if not sa_path.exists():
        raise SystemExit(f"Service account not found: {sa_path}")

    if not firebase_admin._apps:
        cred = credentials.Certificate(str(sa_path))
        firebase_admin.initialize_app(cred)

    db = firestore.client()

    bin_sensor_logs_updated = _cleanup_bin_sensor_logs(db)
    bin_raw_sensor_logs_updated = _cleanup_bin_raw_sensor_logs(db)

    print("Cleanup completed:")
    print(f"- bin_sensor_logs docs updated: {bin_sensor_logs_updated}")
    print(f"- bin_raw_sensor_logs/logs docs updated: {bin_raw_sensor_logs_updated}")


if __name__ == "__main__":
    main()

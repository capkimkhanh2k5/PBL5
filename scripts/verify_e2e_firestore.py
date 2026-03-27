#!/usr/bin/env python3
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore

sa = Path('backend/src/main/resources/serviceAccountKey.json')
if not firebase_admin._apps:
    firebase_admin.initialize_app(credentials.Certificate(str(sa)))

db = firestore.client()

raw_docs = list(
    db.collection('bin_raw_sensor_logs')
    .document('bin_001')
    .collection('logs')
    .order_by('recorded_at', direction=firestore.Query.DESCENDING)
    .limit(1)
    .stream()
)

class_docs = list(
    db.collection('classification_logs')
    .order_by('classified_at', direction=firestore.Query.DESCENDING)
    .limit(1)
    .stream()
)

alert_docs = list(
    db.collection('alerts')
    .order_by('created_at', direction=firestore.Query.DESCENDING)
    .limit(1)
    .stream()
)

print('raw_latest_exists=', bool(raw_docs))
if raw_docs:
    raw = raw_docs[0].to_dict() or {}
    print('raw_latest_keys=', sorted(raw.keys()))
    print('raw_has_temperature=', 'temperature' in raw)

print('classification_latest_exists=', bool(class_docs))
if class_docs:
    c = class_docs[0].to_dict() or {}
    print('classification_latest_bin_id=', c.get('bin_id'))
    print('classification_latest_result=', c.get('classification_result'))
    print('classification_latest_image_url=', c.get('image_url'))

print('alert_latest_exists=', bool(alert_docs))
if alert_docs:
    a = alert_docs[0].to_dict() or {}
    print('alert_latest_bin_id=', a.get('bin_id'))
    print('alert_latest_type=', a.get('alert_type'))
    print('alert_latest_message=', a.get('message'))

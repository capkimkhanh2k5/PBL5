package com.iotSmartTrash.service;

import com.google.cloud.Timestamp;
import com.google.cloud.firestore.DocumentReference;
import com.google.cloud.firestore.DocumentSnapshot;
import com.google.cloud.firestore.Firestore;
import com.google.cloud.firestore.QueryDocumentSnapshot;
import com.iotSmartTrash.exception.ServiceException;
import com.iotSmartTrash.model.Alert;
import com.iotSmartTrash.model.BinRawSensorLog;
import com.iotSmartTrash.model.enums.AlertSeverity;
import com.iotSmartTrash.model.enums.AlertStatus;
import com.iotSmartTrash.model.enums.AlertType;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

@Service
@RequiredArgsConstructor
public class AlertService {

    private static final String COLLECTION_NAME = "alerts";

    private final Firestore firestore;
    private final BinRawSensorLogService rawSensorLogService;
    private final FcmNotificationService fcmNotificationService;

    public String createAlert(Alert alert) {
        try {
            // Snapshot fill-level từ raw log mới nhất (non-realtime mode)
            List<BinRawSensorLog> logs = rawSensorLogService.getRecentLogsForBin(alert.getBinId(), 1);
            if (!logs.isEmpty()) {
                alert.setFillLevelsAtAlert(extractFillLevels(logs.get(0)));
            }

            DocumentReference docRef = firestore.collection(COLLECTION_NAME).document();
            alert.setId(docRef.getId());
            alert.setCreatedAt(Timestamp.now());

            Map<String, Object> payload = new HashMap<>();
            payload.put("id", alert.getId());
            payload.put("bin_id", alert.getBinId());
            payload.put("alert_type", alert.getAlertType() != null ? alert.getAlertType().name() : null);
            payload.put("severity", alert.getSeverity() != null ? alert.getSeverity().name() : null);
            payload.put("message", alert.getMessage());
            payload.put("status", alert.getStatus() != null ? alert.getStatus().name() : AlertStatus.NEW.name());
            payload.put("created_at", alert.getCreatedAt());
            payload.put("resolved_at", alert.getResolvedAt());
            payload.put("resolved_by", alert.getResolvedBy());
            payload.put("fill_levels_at_alert", alert.getFillLevelsAtAlert());
            payload.put("fill_levels_at_resolve", alert.getFillLevelsAtResolve());

            String updateTime = docRef.set(payload).get().getUpdateTime().toString();

            // Send push after successful persistence so Firestore remains source of truth.
            fcmNotificationService.sendAlertCreated(alert);

            return updateTime;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot create alert: operation interrupted", e);
        } catch (ExecutionException e) {
            throw new ServiceException("Cannot create alert", e.getCause());
        }
    }

    public List<Alert> getAllAlerts() {
        try {
            List<Alert> alerts = new ArrayList<>();
            for (QueryDocumentSnapshot doc : firestore.collection(COLLECTION_NAME).get().get().getDocuments()) {
                Alert alert = mapAlert(doc);
                alerts.add(alert);
            }
            return alerts;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot get list of alerts: operation interrupted", e);
        } catch (ExecutionException e) {
            throw new ServiceException("Cannot get list of alerts", e.getCause());
        }
    }

    /**
     * Resolve an alert.
     */
    public String resolveAlert(String alertId, String resolvedByUserId) {
        try {
            DocumentSnapshot alertSnap = firestore.collection(COLLECTION_NAME).document(alertId).get().get();
            if (!alertSnap.exists()) {
                throw new ServiceException("Alert not found");
            }

            String binId = readString(alertSnap, "bin_id", "binId");
            Map<String, Integer> fillLevelsAtResolve = new HashMap<>();

            // Lấy % rác hiện tại (sau khi dọn) để lưu vào lịch sử
            if (binId != null) {
                List<BinRawSensorLog> logs = rawSensorLogService.getRecentLogsForBin(binId, 1);
                if (!logs.isEmpty()) {
                    fillLevelsAtResolve = extractFillLevels(logs.get(0));
                }
            }

            return firestore.collection(COLLECTION_NAME).document(alertId)
                    .update(
                            "status", AlertStatus.RESOLVED.name(),
                            "resolved_by", resolvedByUserId,
                            "resolved_at", Timestamp.now(),
                            "fill_levels_at_resolve", fillLevelsAtResolve)
                    .get()
                    .getUpdateTime()
                    .toString();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot resolve alert: operation interrupted", e);
        } catch (ExecutionException e) {
            throw new ServiceException("Cannot resolve alert: " + alertId, e.getCause());
        }
    }

    private Map<String, Integer> extractFillLevels(BinRawSensorLog log) {
        Map<String, Integer> levels = new HashMap<>();
        levels.put("fillOrganic", safeInt(log.getFillOrganic()));
        levels.put("fillRecycle", safeInt(log.getFillRecycle()));
        levels.put("fillNonRecycle", safeInt(log.getFillNonRecycle()));
        levels.put("fillHazardous", safeInt(log.getFillHazardous()));
        return levels;
    }

    private Alert mapAlert(DocumentSnapshot doc) {
        String id = readString(doc, "id", null);
        if (id == null || id.isBlank()) {
            id = doc.getId();
        }

        return Alert.builder()
                .id(id)
                .binId(readString(doc, "bin_id", "binId"))
                .alertType(readEnum(doc, "alert_type", "alertType", AlertType.class))
                .severity(readEnum(doc, "severity", "severity", AlertSeverity.class))
                .message(readString(doc, "message", "message"))
                .status(readEnum(doc, "status", "status", AlertStatus.class))
                .resolvedBy(readString(doc, "resolved_by", "resolvedBy"))
                .createdAt(readTimestamp(doc, "created_at", "createdAt"))
                .resolvedAt(readTimestamp(doc, "resolved_at", "resolvedAt"))
                .fillLevelsAtAlert(readIntegerMap(doc, "fill_levels_at_alert", "fillLevelsAtAlert"))
                .fillLevelsAtResolve(readIntegerMap(doc, "fill_levels_at_resolve", "fillLevelsAtResolve"))
                .build();
    }

    private String readString(DocumentSnapshot doc, String primary, String fallback) {
        Object value = doc.get(primary);
        if (value == null && fallback != null) {
            value = doc.get(fallback);
        }
        return value != null ? value.toString() : null;
    }

    private Timestamp readTimestamp(DocumentSnapshot doc, String primary, String fallback) {
        Object value = doc.get(primary);
        if (value == null && fallback != null) {
            value = doc.get(fallback);
        }
        if (value instanceof Timestamp timestamp) {
            return timestamp;
        }
        return null;
    }

    private <T extends Enum<T>> T readEnum(DocumentSnapshot doc, String primary, String fallback, Class<T> enumType) {
        String raw = readString(doc, primary, fallback);
        if (raw == null || raw.isBlank()) {
            return null;
        }
        try {
            return Enum.valueOf(enumType, raw);
        } catch (IllegalArgumentException e) {
            return null;
        }
    }

    @SuppressWarnings("unchecked")
    private Map<String, Integer> readIntegerMap(DocumentSnapshot doc, String primary, String fallback) {
        Object value = doc.get(primary);
        if (value == null && fallback != null) {
            value = doc.get(fallback);
        }
        if (!(value instanceof Map<?, ?> source)) {
            return new HashMap<>();
        }
        Map<String, Integer> result = new HashMap<>();
        for (Map.Entry<?, ?> entry : source.entrySet()) {
            String key = entry.getKey() != null ? entry.getKey().toString() : null;
            Object rawValue = entry.getValue();
            if (key == null || !(rawValue instanceof Number number)) {
                continue;
            }
            result.put(key, number.intValue());
        }
        return result;
    }

    private Integer safeInt(Integer value) {
        return value != null ? value : 0;
    }
}

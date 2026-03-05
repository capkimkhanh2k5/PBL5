package com.iotSmartTrash.service;

import com.google.cloud.Timestamp;
import com.google.cloud.firestore.DocumentReference;
import com.google.cloud.firestore.DocumentSnapshot;
import com.google.cloud.firestore.Firestore;
import com.google.cloud.firestore.QueryDocumentSnapshot;
import com.iotSmartTrash.exception.ServiceException;
import com.iotSmartTrash.model.Alert;
import com.iotSmartTrash.model.enums.AlertStatus;
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
    private static final String BINS_REALTIME_COLLECTION = "bins_realtime";

    private final Firestore firestore;

    public String createAlert(Alert alert) {
        try {
            // Lấy thông tin hiện tại của thùng rác để lưu % rác lúc phát cảnh báo
            DocumentSnapshot binSnap = firestore.collection(BINS_REALTIME_COLLECTION)
                    .document(alert.getBinId()).get().get();
            if (binSnap.exists()) {
                alert.setFillLevelsAtAlert(extractFillLevels(binSnap));
            }

            DocumentReference docRef = firestore.collection(COLLECTION_NAME).document();
            alert.setId(docRef.getId());
            alert.setCreatedAt(Timestamp.now());
            return docRef.set(alert).get().getUpdateTime().toString();
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
                Alert alert = doc.toObject(Alert.class);
                alert.setId(doc.getId());
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

            String binId = alertSnap.getString("bin_id");
            Map<String, Integer> fillLevelsAtResolve = new HashMap<>();

            // Lấy % rác hiện tại (sau khi dọn) để lưu vào lịch sử
            if (binId != null) {
                DocumentSnapshot binSnap = firestore.collection(BINS_REALTIME_COLLECTION)
                        .document(binId).get().get();
                if (binSnap.exists()) {
                    fillLevelsAtResolve = extractFillLevels(binSnap);
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

    private Map<String, Integer> extractFillLevels(DocumentSnapshot snap) {
        Map<String, Integer> levels = new HashMap<>();
        levels.put("fillOrganic", getIntValue(snap, "fill_organic"));
        levels.put("fillRecycle", getIntValue(snap, "fill_recycle"));
        levels.put("fillNonRecycle", getIntValue(snap, "fill_non_recycle"));
        levels.put("fillHazardous", getIntValue(snap, "fill_hazardous"));
        return levels;
    }

    private Integer getIntValue(DocumentSnapshot snap, String field) {
        Long val = snap.getLong(field);
        return val != null ? val.intValue() : 0;
    }
}

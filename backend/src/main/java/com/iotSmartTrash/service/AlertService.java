package com.iotSmartTrash.service;

import com.google.cloud.Timestamp;
import com.google.cloud.firestore.DocumentReference;
import com.google.cloud.firestore.Firestore;
import com.google.cloud.firestore.QueryDocumentSnapshot;
import com.iotSmartTrash.exception.ServiceException;
import com.iotSmartTrash.model.Alert;
import com.iotSmartTrash.model.enums.AlertStatus;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

@Service
@RequiredArgsConstructor
public class AlertService {

    private static final String COLLECTION_NAME = "alerts";

    private final Firestore firestore;

    public String createAlert(Alert alert) {
        try {
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
            return firestore.collection(COLLECTION_NAME).document(alertId)
                    .update(
                            "status", AlertStatus.RESOLVED.name(),
                            "resolved_by", resolvedByUserId,
                            "resolved_at", Timestamp.now())
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
}

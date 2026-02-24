package com.iotSmartTrash.service;

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
import java.util.Map;

@Service
@RequiredArgsConstructor
public class AlertService {

    private static final String COLLECTION_NAME = "alerts";

    private final Firestore firestore;

    /** Tạo cảnh báo mới (do Raspberry Pi gọi API hoặc Cron Job tự chạy) */
    public String createAlert(Alert alert) {
        try {
            DocumentReference docRef = firestore.collection(COLLECTION_NAME).document();
            alert.setId(docRef.getId());
            alert.setCreatedAt(com.google.cloud.Timestamp.now());
            // status đã được set = AlertStatus.NEW trong AlertCreateDTO.toModel()

            return docRef.set(alert).get().getUpdateTime().toString();
        } catch (Exception e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot create alert", e);
        }
    }

    /** Lấy danh sách toàn bộ cảnh báo cho Admin Portal */
    public List<Alert> getAllAlerts() {
        try {
            List<Alert> alerts = new ArrayList<>();
            for (QueryDocumentSnapshot doc : firestore.collection(COLLECTION_NAME).get().get().getDocuments()) {
                Alert alert = doc.toObject(Alert.class);
                alert.setId(doc.getId());
                alerts.add(alert);
            }
            return alerts;
        } catch (Exception e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot get list of alerts", e);
        }
    }

    /** Staff/Admin xác nhận đã xử lý xong cảnh báo */
    public String resolveAlert(String alertId, String resolvedByUserId) {
        try {
            return firestore.collection(COLLECTION_NAME).document(alertId)
                    .update(Map.of(
                            "status", AlertStatus.RESOLVED.name(),
                            "resolved_by", resolvedByUserId,
                            "resolved_at", com.google.cloud.Timestamp.now()))
                    .get()
                    .getUpdateTime()
                    .toString();
        } catch (Exception e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot resolve alert: " + alertId, e);
        }
    }
}

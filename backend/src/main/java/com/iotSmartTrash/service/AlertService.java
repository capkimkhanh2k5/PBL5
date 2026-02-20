package com.iotSmartTrash.service;

import com.google.api.core.ApiFuture;
import com.google.cloud.firestore.DocumentReference;
import com.google.cloud.firestore.Firestore;
import com.google.cloud.firestore.QueryDocumentSnapshot;
import com.google.cloud.firestore.WriteResult;
import com.google.firebase.cloud.FirestoreClient;
import com.iotSmartTrash.model.Alert;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

@Service
public class AlertService {

    private static final String COLLECTION_NAME = "alerts";

    /**
     * Dùng cho việc tạo cảnh báo (do Raspberry Pi gọi API hoặc Cron Job tự chạy)
     */
    public String createAlert(Alert alert) throws ExecutionException, InterruptedException {
        Firestore dbFirestore = FirestoreClient.getFirestore();
        DocumentReference docRef = dbFirestore.collection(COLLECTION_NAME).document();

        alert.setId(docRef.getId());
        alert.setStatus("NEW");
        alert.setCreated_at(com.google.cloud.Timestamp.now());

        ApiFuture<WriteResult> collectionsApiFuture = docRef.set(alert);
        return collectionsApiFuture.get().getUpdateTime().toString();
    }

    /**
     * Lấy danh sách Toàn bộ cảnh báo cho Dashboard Web/App Admin
     */
    public List<Alert> getAllAlerts() throws ExecutionException, InterruptedException {
        Firestore dbFirestore = FirestoreClient.getFirestore();
        List<Alert> alerts = new ArrayList<>();

        for (QueryDocumentSnapshot document : dbFirestore.collection(COLLECTION_NAME).get().get().getDocuments()) {
            Alert alert = document.toObject(Alert.class);
            alert.setId(document.getId());
            alerts.add(alert);
        }
        return alerts;
    }

    /**
     * Staff/Admin xác nhận đã xử lý xong cảnh báo
     */
    public String resolveAlert(String alertId, String resolvedByUserId)
            throws ExecutionException, InterruptedException {
        Firestore dbFirestore = FirestoreClient.getFirestore();
        ApiFuture<WriteResult> updateFuture = dbFirestore.collection(COLLECTION_NAME).document(alertId)
                .update(
                        "status", "RESOLVED",
                        "resolved_by", resolvedByUserId,
                        "resolved_at", com.google.cloud.Timestamp.now());
        return updateFuture.get().getUpdateTime().toString();
    }
}

package com.iotSmartTrash.service;

import com.google.firebase.messaging.FirebaseMessaging;
import com.google.firebase.messaging.Message;
import com.google.firebase.messaging.Notification;
import com.iotSmartTrash.model.Alert;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;

@Service
@Slf4j
public class FcmNotificationService {

    private static final String ALERT_TOPIC = "system-alerts";

    public void sendAlertCreated(Alert alert) {
        if (alert == null) {
            return;
        }

        String title = "Canh bao he thong";
        String body = buildBody(alert);

        Map<String, String> data = new HashMap<>();
        data.put("eventType", "ALERT_CREATED");
        data.put("alertId", safe(alert.getId()));
        data.put("binId", safe(alert.getBinId()));
        data.put("alertType", alert.getAlertType() != null ? alert.getAlertType().name() : "");
        data.put("severity", alert.getSeverity() != null ? alert.getSeverity().name() : "");
        data.put("message", safe(alert.getMessage()));

        Message message = Message.builder()
                .setTopic(ALERT_TOPIC)
                .setNotification(Notification.builder()
                        .setTitle(title)
                        .setBody(body)
                        .build())
                .putAllData(data)
                .build();

        try {
            FirebaseMessaging.getInstance().send(message);
        } catch (Exception e) {
            log.error("Failed to send FCM alert notification for alertId={}: {}", alert.getId(), e.getMessage(), e);
        }
    }

    private String buildBody(Alert alert) {
        String severity = alert.getSeverity() != null ? alert.getSeverity().name() : "UNKNOWN";
        String alertType = alert.getAlertType() != null ? alert.getAlertType().name() : "UNKNOWN";
        String msg = safe(alert.getMessage());
        if (msg.isBlank()) {
            return "Bin " + safe(alert.getBinId()) + " - " + alertType + " (" + severity + ")";
        }
        return "Bin " + safe(alert.getBinId()) + " - " + msg;
    }

    private String safe(String value) {
        return value != null ? value : "";
    }
}

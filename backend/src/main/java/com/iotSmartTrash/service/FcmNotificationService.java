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

        String title = buildTitle(alert);
        String body = buildBody(alert);

        Map<String, String> data = new HashMap<>();
        data.put("eventType", "ALERT_CREATED");
        data.put("alertId", safe(alert.getId()));
        data.put("binId", safe(alert.getBinId()));
        data.put("alertType", alert.getAlertType() != null ? alert.getAlertType().name() : "");
        data.put("severity", alert.getSeverity() != null ? alert.getSeverity().name() : "");
        data.put("message", body);

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
            log.info("Sent alert notification for alertId={}, binId={}",
                    alert.getId(), alert.getBinId());
        } catch (Exception e) {
            log.error("Failed to send FCM alert notification for alertId={}: {}",
                    alert.getId(), e.getMessage(), e);
        }
    }

    public void sendBinOnlineAgain(String binId) {
        String safeBinId = safe(binId);

        String title = "SmartBin Back Online";
        String body = "Bin " + safeBinId + " is online .";

        Map<String, String> data = new HashMap<>();
        data.put("eventType", "BIN_ONLINE_AGAIN");
        data.put("binId", safeBinId);
        data.put("alertType", "OFFLINE");
        data.put("status", "RESOLVED");
        data.put("message", body);

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
            log.info("Sent BIN_ONLINE_AGAIN notification for binId={}", safeBinId);
        } catch (Exception e) {
            log.error("Failed to send BIN_ONLINE_AGAIN notification for binId={}: {}",
                    safeBinId, e.getMessage(), e);
        }
    }

    private String buildTitle(Alert alert) {
        if (alert.getAlertType() == null) {
            return "SmartBin Alert";
        }

        return switch (alert.getAlertType()) {
            case OFFLINE -> "SmartBin Offline Alert";
            default -> "SmartBin System Alert";
        };
    }

    private String buildBody(Alert alert) {
        String binId = safe(alert.getBinId());
        String msg = safe(alert.getMessage());

        if (alert.getAlertType() != null) {
            switch (alert.getAlertType()) {
                case OFFLINE:
                    return "Bin " + binId + " is offline.";
                default:
                    break;
            }
        }

        if (!msg.isBlank()) {
            return "Bin " + binId + " - " + msg;
        }

        String severity = alert.getSeverity() != null ? alert.getSeverity().name() : "UNKNOWN";
        String alertType = alert.getAlertType() != null ? alert.getAlertType().name() : "UNKNOWN";

        return "Bin " + binId + " - " + alertType + " alert detected. Severity: " + severity + ".";
    }

    private String safe(String value) {
        return value != null ? value : "";
    }
}
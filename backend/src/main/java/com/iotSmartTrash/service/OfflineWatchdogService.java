package com.iotSmartTrash.service;

import com.google.cloud.Timestamp;
import com.google.cloud.firestore.DocumentChange;
import com.google.cloud.firestore.DocumentSnapshot;
import com.google.cloud.firestore.Firestore;
import com.google.cloud.firestore.ListenerRegistration;
import com.google.cloud.firestore.QuerySnapshot;
import com.iotSmartTrash.model.Alert;
import com.iotSmartTrash.model.enums.AlertSeverity;
import com.iotSmartTrash.model.enums.AlertStatus;
import com.iotSmartTrash.model.enums.AlertType;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

@Service
@RequiredArgsConstructor
@Slf4j
public class OfflineWatchdogService {

    private static final String RAW_LOG_PARENT_COLLECTION = "bin_raw_sensor_logs";
    private static final String RAW_LOG_SUB_COLLECTION = "logs";

    private final Firestore firestore;
    private final AlertService alertService;
    private final FcmNotificationService fcmNotificationService;

    // Mặc định: 5 phút 30 giây
    @Value("${smartbin.offline-after-ms:330000}")
    private long offlineAfterMs;

    @Value("${smartbin.offline-listener-enabled:true}")
    private boolean listenerEnabled;

    private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(2);

    private final Map<String, ScheduledFuture<?>> timersByBinId = new ConcurrentHashMap<>();
    private final Map<String, Long> lastSeenMillisByBinId = new ConcurrentHashMap<>();

    // Cache cục bộ để tránh query alerts liên tục khi bin vẫn online
    private final Set<String> offlineBinIds = ConcurrentHashMap.newKeySet();

    private ListenerRegistration listenerRegistration;

    @PostConstruct
    public void start() {
        if (!listenerEnabled) {
            log.info("Offline watchdog listener is disabled.");
            return;
        }

        listenerRegistration = firestore
                .collectionGroup(RAW_LOG_SUB_COLLECTION)
                .addSnapshotListener((snapshots, error) -> {
                    if (error != null) {
                        log.error("Offline watchdog Firestore listener error: {}", error.getMessage(), error);
                        return;
                    }

                    if (snapshots == null) {
                        return;
                    }

                    handleSnapshot(snapshots);
                });

        log.info("Offline watchdog listener started. offlineAfterMs={} ms", offlineAfterMs);
    }

    @PreDestroy
    public void stop() {
        if (listenerRegistration != null) {
            listenerRegistration.remove();
        }

        for (ScheduledFuture<?> timer : timersByBinId.values()) {
            timer.cancel(false);
        }

        scheduler.shutdownNow();
        log.info("Offline watchdog listener stopped.");
    }

    private void handleSnapshot(QuerySnapshot snapshots) {
        Map<String, Long> latestMillisInThisBatch = new HashMap<>();

        for (DocumentChange change : snapshots.getDocumentChanges()) {
            if (change.getType() == DocumentChange.Type.REMOVED) {
                continue;
            }

            DocumentSnapshot doc = change.getDocument();

            String binId = extractBinId(doc);
            if (binId == null || binId.isBlank()) {
                continue;
            }

            Long recordedAtMillis = readRecordedAtMillis(doc);
            if (recordedAtMillis == null || recordedAtMillis <= 0) {
                continue;
            }

            latestMillisInThisBatch.merge(binId, recordedAtMillis, Math::max);
        }

        latestMillisInThisBatch.forEach(this::registerLatestLog);
    }

    private void registerLatestLog(String binId, long recordedAtMillis) {
        Long current = lastSeenMillisByBinId.get(binId);

        // Nếu log cũ hơn log đã biết thì bỏ qua
        if (current != null && recordedAtMillis < current) {
            return;
        }

        lastSeenMillisByBinId.put(binId, recordedAtMillis);

        long diffMs = System.currentTimeMillis() - recordedAtMillis;

        // Nếu trước đó đã offline, nay có log mới thì resolve alert
        if (diffMs < offlineAfterMs) {
            try {
                boolean wasMarkedOffline = offlineBinIds.remove(binId);
                boolean hasActiveOfflineAlert =
                        wasMarkedOffline || alertService.hasActiveOfflineAlert(binId);

                if (hasActiveOfflineAlert) {
                    alertService.resolveOfflineAlert(binId);

                    fcmNotificationService.sendBinOnlineAgain(binId);

                    log.info("[ONLINE AGAIN] binId={}, lastLogAgeSeconds={}",
                            binId, diffMs / 1000);
                }

            } catch (Exception e) {
                log.error("Cannot resolve offline alert for binId={}: {}",
                        binId, e.getMessage(), e);
            }
        }

        resetTimer(binId, recordedAtMillis);
    }

    private void resetTimer(String binId, long expectedLastSeenMillis) {
        ScheduledFuture<?> oldTimer = timersByBinId.remove(binId);
        if (oldTimer != null) {
            oldTimer.cancel(false);
        }

        long fireAtMillis = expectedLastSeenMillis + offlineAfterMs;
        long delayMs = Math.max(0L, fireAtMillis - System.currentTimeMillis());

        ScheduledFuture<?> newTimer = scheduler.schedule(
                () -> checkOffline(binId, expectedLastSeenMillis),
                delayMs,
                TimeUnit.MILLISECONDS
        );

        timersByBinId.put(binId, newTimer);

        log.info(
                "[OfflineWatchdog] binId={}, lastSeen={}, checkAfterSeconds={}",
                binId,
                expectedLastSeenMillis,
                delayMs / 1000
        );
    }

    private void checkOffline(String binId, long expectedLastSeenMillis) {
        Long actualLastSeen = lastSeenMillisByBinId.get(binId);

        // Nếu đã có log mới hơn rồi thì timer cũ không còn giá trị
        if (actualLastSeen == null || actualLastSeen != expectedLastSeenMillis) {
            return;
        }

        long diffMs = System.currentTimeMillis() - actualLastSeen;

        if (diffMs < offlineAfterMs) {
            resetTimer(binId, actualLastSeen);
            return;
        }

        try {
            // Check Firestore 1 lần để tránh tạo trùng alert nếu đã có OFFLINE NEW
            if (!alertService.hasActiveOfflineAlert(binId)) {
                Alert alert = Alert.builder()
                        .binId(binId)
                        .alertType(AlertType.OFFLINE)
                        .severity(AlertSeverity.WARNING)
                        .message("Bin is offline.")
                        .status(AlertStatus.NEW)
                        .build();

                alertService.createAlert(alert);
            }

            offlineBinIds.add(binId);

            log.warn("🔴 OFFLINE DETECTED: binId={}, lastLogAgeSeconds={}", binId, diffMs / 1000);

        } catch (Exception e) {
            log.error("Cannot create offline alert for binId={}: {}", binId, e.getMessage(), e);
        }
    }

    private String extractBinId(DocumentSnapshot doc) {
        String path = doc.getReference().getPath();

        // Path đúng:
        // bin_raw_sensor_logs/{binId}/logs/{autoId}
        String[] parts = path.split("/");

        if (parts.length >= 4
                && RAW_LOG_PARENT_COLLECTION.equals(parts[0])
                && RAW_LOG_SUB_COLLECTION.equals(parts[2])) {
            return parts[1];
        }

        return null;
    }

    private Long readRecordedAtMillis(DocumentSnapshot doc) {
        Object value = doc.get("recordedAt");

        if (value == null) {
            value = doc.get("recorded_at");
        }

        if (value instanceof Timestamp timestamp) {
            return timestamp.toDate().getTime();
        }

        if (value instanceof Number number) {
            return number.longValue();
        }

        return null;
    }
}
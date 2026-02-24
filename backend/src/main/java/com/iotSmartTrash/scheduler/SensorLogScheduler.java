package com.iotSmartTrash.scheduler;

import com.google.cloud.firestore.Firestore;
import com.google.firebase.database.*;
import com.iotSmartTrash.model.BinSensorLog;
import com.iotSmartTrash.model.enums.BinPeriod;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;

@Component // ← Fix: đổi từ @Service sang @Component (Scheduler không phải Service)
@Slf4j
@RequiredArgsConstructor
public class SensorLogScheduler {

    private static final long LOG_TTL_MS = 24 * 60 * 60 * 1000L; // 24 giờ

    private final Firestore firestore; // ← Fix: inject qua DI thay vì FirestoreClient.getFirestore()

    /**
     * Chạy mỗi 6 tiếng (00:00, 06:00, 12:00, 18:00) — tổng hợp RTDB → Firestore
     */
    @Scheduled(cron = "0 0 0,6,12,18 * * *")
    public void aggregateSensorLogs() {
        ZonedDateTime now = ZonedDateTime.now(ZoneOffset.UTC);
        BinPeriod period = BinPeriod.fromHour(now.getHour());
        String date = now.format(DateTimeFormatter.ISO_LOCAL_DATE);

        log.info("[Scheduler] Start aggregate Sensor Logs — period={}, date={}", period, date);

        DatabaseReference rtdbRef = FirebaseDatabase.getInstance().getReference("bin_sensor_logs");
        CountDownLatch latch = new CountDownLatch(1);

        rtdbRef.addListenerForSingleValueEvent(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot snapshot) {
                if (!snapshot.exists()) {
                    log.info("[Scheduler] Not found data in Realtime DB.");
                    latch.countDown();
                    return;
                }

                for (DataSnapshot binSnap : snapshot.getChildren()) {
                    String binId = binSnap.getKey();
                    List<Map<String, Object>> logs = new ArrayList<>();

                    for (DataSnapshot logSnap : binSnap.getChildren()) {
                        @SuppressWarnings("unchecked")
                        Map<String, Object> logData = (Map<String, Object>) logSnap.getValue();
                        if (logData != null)
                            logs.add(logData);
                    }

                    if (logs.isEmpty())
                        continue;

                    BinSensorLog stats = computeStats(binId, logs, period, date);
                    String docId = binId + "_" + date + "_" + period.name();

                    firestore.collection("bin_sensor_logs").document(docId).set(stats);
                    log.info("[Scheduler] Saved 6h aggregate for bin: {}", binId);
                }
                latch.countDown();
            }

            @Override
            public void onCancelled(DatabaseError error) {
                log.error("[Scheduler] Error reading Realtime DB: ", error.toException());
                latch.countDown();
            }
        });

        try {
            latch.await();
            log.info("[Scheduler] Completed aggregateSensorLogs.");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            log.warn("[Scheduler] aggregateSensorLogs bị interrupted.");
        }
    }

    /**
     * Dọn dẹp log cũ hơn 24h trong Realtime DB. Chạy 30 phút sau aggregate.
     */
    @Scheduled(cron = "0 30 0,6,12,18 * * *")
    public void cleanupRealtimeLogs() {
        long cutoff = System.currentTimeMillis() - LOG_TTL_MS;
        log.info("[Scheduler] Start cleanup old logs before {}", cutoff);

        DatabaseReference rtdbRef = FirebaseDatabase.getInstance().getReference("bin_sensor_logs");
        CountDownLatch latch = new CountDownLatch(1);

        rtdbRef.addListenerForSingleValueEvent(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot snapshot) {
                if (!snapshot.exists()) {
                    latch.countDown();
                    return;
                }

                int deletedCount = 0;
                for (DataSnapshot binSnap : snapshot.getChildren()) {
                    for (DataSnapshot logSnap : binSnap.getChildren()) {
                        @SuppressWarnings("unchecked")
                        Map<String, Object> logData = (Map<String, Object>) logSnap.getValue();
                        if (logData != null && logData.containsKey("recorded_at")) {
                            long recordedAt = ((Number) logData.get("recorded_at")).longValue();
                            if (recordedAt < cutoff) {
                                logSnap.getRef().removeValueAsync();
                                deletedCount++;
                            }
                        }
                    }
                }
                log.info("[Scheduler] Deleted {} old logs.", deletedCount);
                latch.countDown();
            }

            @Override
            public void onCancelled(DatabaseError error) {
                log.error("[Scheduler] Error cleanup: ", error.toException());
                latch.countDown();
            }
        });

        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    // ── Helper methods ─────────────────────────────────────────

    private BinSensorLog computeStats(String binId, List<Map<String, Object>> logs,
            BinPeriod period, String date) {
        double sumTemp = 0, minTemp = Double.MAX_VALUE, maxTemp = Double.MIN_VALUE;
        long sumBattery = 0, sumOrg = 0, sumRec = 0, sumNon = 0, sumHaz = 0;

        for (Map<String, Object> log : logs) {
            double temp = getDouble(log, "temperature");
            sumTemp += temp;
            if (temp < minTemp)
                minTemp = temp;
            if (temp > maxTemp)
                maxTemp = temp;

            sumBattery += getLong(log, "battery_level");
            sumOrg += getLong(log, "fill_organic");
            sumRec += getLong(log, "fill_recycle");
            sumNon += getLong(log, "fill_non_recycle");
            sumHaz += getLong(log, "fill_hazardous");
        }

        int n = logs.size();
        return BinSensorLog.builder()
                .binId(binId)
                .period(period)
                .date(date)
                .avgTemperature(Math.round((sumTemp / n) * 10.0) / 10.0)
                .minTemperature(minTemp)
                .maxTemperature(maxTemp)
                .avgBattery((int) (sumBattery / n))
                .avgFillOrganic((int) (sumOrg / n))
                .avgFillRecycle((int) (sumRec / n))
                .avgFillNonRecycle((int) (sumNon / n))
                .avgFillHazardous((int) (sumHaz / n))
                .sampleCount(n)
                .recordedAt(com.google.cloud.Timestamp.now())
                .build();
    }

    private double getDouble(Map<String, Object> map, String key) {
        Object val = map.get(key);
        return val instanceof Number ? ((Number) val).doubleValue() : 0.0;
    }

    private long getLong(Map<String, Object> map, String key) {
        Object val = map.get(key);
        return val instanceof Number ? ((Number) val).longValue() : 0L;
    }
}

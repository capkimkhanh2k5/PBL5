package com.iotSmartTrash.scheduler;

import com.google.cloud.firestore.Firestore;
import com.google.firebase.cloud.FirestoreClient;
import com.google.firebase.database.*;
import com.iotSmartTrash.model.BinSensorLog;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;

@Service
@Slf4j
@RequiredArgsConstructor
public class SensorLogScheduler {

    private static final long LOG_TTL_MS = 24 * 60 * 60 * 1000L; // 24 giờ

    /**
     * Chạy mỗi 6 tiếng một lần (00:00, 06:00, 12:00, 18:00)
     * Aggregate (tổng hợp) 1 loạt dữ liệu từ Realtime DB sang Firestore
     */
    @Scheduled(cron = "0 0 0,6,12,18 * * *")
    public void aggregateSensorLogs() {
        ZonedDateTime now = ZonedDateTime.now(ZoneOffset.UTC);
        String period = getPeriodLabel(now.getHour());
        String date = now.format(DateTimeFormatter.ISO_LOCAL_DATE);

        log.info("[Scheduler] Start aggregate Sensor Logs for kỳ {} ngày {}", period, date);

        DatabaseReference rtdbRef = FirebaseDatabase.getInstance().getReference("bin_sensor_logs");
        Firestore dbFirestore = FirestoreClient.getFirestore();

        // Sử dụng CountDownLatch để chờ Firebase Async callback hoàn thành
        CountDownLatch latch = new CountDownLatch(1);

        rtdbRef.addListenerForSingleValueEvent(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot snapshot) {
                if (!snapshot.exists()) {
                    log.info("Not found data in Realtime DB.");
                    latch.countDown();
                    return;
                }

                for (DataSnapshot binSnap : snapshot.getChildren()) {
                    String binId = binSnap.getKey();
                    List<Map<String, Object>> logs = new ArrayList<>();

                    for (DataSnapshot logSnap : binSnap.getChildren()) {
                        logs.add((Map<String, Object>) logSnap.getValue());
                    }

                    if (logs.isEmpty())
                        continue;

                    BinSensorLog stats = computeStats(binId, logs, period, date);
                    String docId = binId + "_" + date + "_" + period;

                    // Lưu vào Firestore
                    dbFirestore.collection("bin_sensor_logs").document(docId).set(stats);
                    log.info("Saved 6 hours aggregate for bin: {}", binId);
                }
                latch.countDown();
            }

            @Override
            public void onCancelled(DatabaseError error) {
                log.error("Error reading Realtime DB: ", error.toException());
                latch.countDown();
            }
        });

        try {
            latch.await();
            log.info("[Scheduler] Completed aggregateSensorLogs.");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /**
     * Dọn dẹp dữ liệu cũ hơn 24h trong Realtime DB. Chạy 30 phút sau khi Aggregate
     * xong.
     */
    @Scheduled(cron = "0 30 0,6,12,18 * * *")
    public void cleanupRealtimeLogs() {
        long cutoff = System.currentTimeMillis() - LOG_TTL_MS;
        log.info("[Scheduler] Start cleanup old logs {}", cutoff);

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
                log.error("Error: ", error.toException());
                latch.countDown();
            }
        });

        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    // Helper Functions
    private String getPeriodLabel(int hour) {
        if (hour < 6)
            return "00h";
        if (hour < 12)
            return "06h";
        if (hour < 18)
            return "12h";
        return "18h";
    }

    private BinSensorLog computeStats(String binId, List<Map<String, Object>> logs, String period, String date) {
        double sumTemp = 0;
        double minTemp = Double.MAX_VALUE;
        double maxTemp = Double.MIN_VALUE;

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
                .bin_id(binId)
                .period(period)
                .date(date)
                .avg_temperature(Math.round((sumTemp / n) * 10.0) / 10.0)
                .min_temperature(minTemp)
                .max_temperature(maxTemp)
                .avg_battery((int) (sumBattery / n))
                .avg_fill_organic((int) (sumOrg / n))
                .avg_fill_recycle((int) (sumRec / n))
                .avg_fill_non_recycle((int) (sumNon / n))
                .avg_fill_hazardous((int) (sumHaz / n))
                .sample_count(n)
                .recorded_at(com.google.cloud.Timestamp.now())
                .build();
    }

    private double getDouble(Map<String, Object> map, String key) {
        Object val = map.get(key);
        if (val instanceof Number)
            return ((Number) val).doubleValue();
        return 0.0;
    }

    private long getLong(Map<String, Object> map, String key) {
        Object val = map.get(key);
        if (val instanceof Number)
            return ((Number) val).longValue();
        return 0L;
    }
}

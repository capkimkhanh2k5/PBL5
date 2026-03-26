package com.iotSmartTrash.scheduler;

import com.google.cloud.Timestamp;
import com.google.cloud.firestore.Firestore;
import com.iotSmartTrash.model.BinRawSensorLog;
import com.iotSmartTrash.model.BinSensorLog;
import com.iotSmartTrash.model.enums.BinPeriod;
import com.iotSmartTrash.service.BinRawSensorLogService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;

@Component
@Slf4j
@RequiredArgsConstructor
public class SensorLogScheduler {

    private static final long LOG_TTL_MS = 24 * 60 * 60 * 1000L; // 24 giờ

    private final Firestore firestore;
    private final BinRawSensorLogService rawSensorLogService;

    /**
     * Chạy mỗi 6 tiếng (00:00, 06:00, 12:00, 18:00).
     * Đọc raw sensor logs từ Firestore subcollection, tổng hợp → Firestore collection bin_sensor_logs.
     */
    @Scheduled(cron = "0 0 0,6,12,18 * * *")
    public void aggregateSensorLogs() {
        ZonedDateTime now = ZonedDateTime.now(ZoneId.systemDefault());
        BinPeriod period = BinPeriod.fromHour(now.getHour());
        String date = now.format(DateTimeFormatter.ISO_LOCAL_DATE);

        log.info("[Scheduler] Start aggregate Sensor Logs — period={}, date={}", period, date);

        try {
            List<String> binIds = rawSensorLogService.getAllBinIds();

            if (binIds.isEmpty()) {
                log.info("[Scheduler] No bin data found in Firestore raw sensor logs.");
                return;
            }

            for (String binId : binIds) {
                List<BinRawSensorLog> logs = rawSensorLogService.getLogsForBin(binId);

                if (logs.isEmpty()) {
                    continue;
                }

                BinSensorLog stats = computeStats(binId, logs, period, date);
                String docId = binId + "_" + date + "_" + period.toLabel();

                firestore.collection("bin_sensor_logs").document(docId).set(stats);
                log.info("[Scheduler] Saved 6h aggregate for bin: {} ({} samples)", binId, logs.size());
            }

            log.info("[Scheduler] Completed aggregateSensorLogs.");
        } catch (Exception e) {
            log.error("[Scheduler] Error during aggregateSensorLogs: {}", e.getMessage(), e);
        }
    }

    /**
     * Dọn dẹp raw sensor logs cũ hơn 24h trong Firestore. Chạy 30 phút sau aggregate.
     */
    @Scheduled(cron = "0 30 0,6,12,18 * * *")
    public void cleanupOldSensorLogs() {
        long cutoff = System.currentTimeMillis() - LOG_TTL_MS;
        log.info("[Scheduler] Start cleanup old raw sensor logs before {}", cutoff);

        try {
            List<String> binIds = rawSensorLogService.getAllBinIds();
            int totalDeleted = 0;

            for (String binId : binIds) {
                int deleted = rawSensorLogService.deleteOldLogs(binId, cutoff);
                totalDeleted += deleted;
            }

            log.info("[Scheduler] Cleanup completed. Deleted {} old raw sensor logs.", totalDeleted);
        } catch (Exception e) {
            log.error("[Scheduler] Error during cleanup: {}", e.getMessage(), e);
        }
    }

    // ── Helper methods ─────────────────────────────────────────

    private BinSensorLog computeStats(String binId, List<BinRawSensorLog> logs,
            BinPeriod period, String date) {
        long sumBattery = 0, sumOrg = 0, sumRec = 0, sumNon = 0, sumHaz = 0;

        for (BinRawSensorLog sensorLog : logs) {
            sumBattery += sensorLog.getBatteryLevel() != null ? sensorLog.getBatteryLevel() : 0;
            sumOrg += sensorLog.getFillOrganic() != null ? sensorLog.getFillOrganic() : 0;
            sumRec += sensorLog.getFillRecycle() != null ? sensorLog.getFillRecycle() : 0;
            sumNon += sensorLog.getFillNonRecycle() != null ? sensorLog.getFillNonRecycle() : 0;
            sumHaz += sensorLog.getFillHazardous() != null ? sensorLog.getFillHazardous() : 0;
        }

        int n = logs.size();
        return BinSensorLog.builder()
                .binId(binId)
                .period(period)
                .date(date)
                .avgBattery((int) (sumBattery / n))
                .avgFillOrganic((int) (sumOrg / n))
                .avgFillRecycle((int) (sumRec / n))
                .avgFillNonRecycle((int) (sumNon / n))
                .avgFillHazardous((int) (sumHaz / n))
                .sampleCount(n)
                .recordedAt(Timestamp.now())
                .build();
    }
}

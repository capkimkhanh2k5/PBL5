package com.iotSmartTrash.scheduler;

import com.google.cloud.Timestamp;
import com.google.cloud.firestore.Firestore;
import com.iotSmartTrash.model.BinRawSensorLog;
import com.iotSmartTrash.service.BinRawSensorLogService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.time.LocalDate;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Component
@Slf4j
@RequiredArgsConstructor
public class DailyStatisticsScheduler {

    private static final String DAILY_COLLECTION = "bin_daily_statistics";
    private static final ZoneId ZONE = ZoneId.of("Asia/Ho_Chi_Minh");

    private final Firestore firestore;
    private final BinRawSensorLogService rawSensorLogService;

    @Value("${smartbin.compartment-capacity-liters}")
    private double compartmentCapacityLiters;

    /**
     * Chạy lúc 00:05 mỗi ngày.
     * Tổng hợp dữ liệu của ngày hôm qua.
     */
    @Scheduled(cron = "0 5 0 * * *", zone = "Asia/Ho_Chi_Minh")
    public void aggregateYesterdayDailyStatistics() {
        LocalDate targetDate = LocalDate.now(ZONE).minusDays(1);
        aggregateDailyStatisticsForDate(targetDate);
    }

    public void aggregateDailyStatisticsForDate(LocalDate date) {
        log.info("[DailyStats] Start aggregate daily statistics for date={}", date);

        try {
            ZonedDateTime startOfDay = date.atStartOfDay(ZONE);
            ZonedDateTime endOfDay = startOfDay.plusDays(1);

            Timestamp startTimestamp = Timestamp.ofTimeSecondsAndNanos(
                    startOfDay.toEpochSecond(),
                    0
            );

            Timestamp endTimestamp = Timestamp.ofTimeSecondsAndNanos(
                    endOfDay.toEpochSecond(),
                    0
            );

            List<String> binIds = rawSensorLogService.getAllBinIds();

            if (binIds.isEmpty()) {
                log.info("[DailyStats] No bins found.");
                return;
            }

            for (String binId : binIds) {
                List<BinRawSensorLog> logs = rawSensorLogService.getLogsForBinBetween(
                        binId,
                        startTimestamp,
                        endTimestamp
                );

                if (logs.isEmpty()) {
                    log.info("[DailyStats] No raw logs for bin={}, date={}", binId, date);
                    continue;
                }

                Map<String, Object> dailyStats = computeDailyStats(binId, date, logs);

                String docId = binId + "_" + date;

                firestore
                        .collection(DAILY_COLLECTION)
                        .document(docId)
                        .set(dailyStats)
                        .get();

                log.info(
                        "[DailyStats] Saved daily stats: bin={}, date={}, samples={}",
                        binId,
                        date,
                        logs.size()
                );
            }

            log.info("[DailyStats] Completed aggregate daily statistics for date={}", date);

        } catch (Exception e) {
            log.error("[DailyStats] Error aggregate daily statistics: {}", e.getMessage(), e);
        }
    }

    private Map<String, Object> computeDailyStats(
            String binId,
            LocalDate date,
            List<BinRawSensorLog> logs
    ) {
        long sumOrganic = 0;
        long sumRecycle = 0;
        long sumNonRecycle = 0;
        long sumHazardous = 0;

        for (BinRawSensorLog log : logs) {
            sumOrganic += safeInt(log.getFillOrganic());
            sumRecycle += safeInt(log.getFillRecycle());
            sumNonRecycle += safeInt(log.getFillNonRecycle());
            sumHazardous += safeInt(log.getFillHazardous());
        }

        int sampleCount = logs.size();

        double avgOrganic = sumOrganic / (double) sampleCount;
        double avgRecycle = sumRecycle / (double) sampleCount;
        double avgNonRecycle = sumNonRecycle / (double) sampleCount;
        double avgHazardous = sumHazardous / (double) sampleCount;

        double organicLiters = avgOrganic / 100.0 * compartmentCapacityLiters;
        double recycleLiters = avgRecycle / 100.0 * compartmentCapacityLiters;
        double nonRecycleLiters = avgNonRecycle / 100.0 * compartmentCapacityLiters;
        double hazardousLiters = avgHazardous / 100.0 * compartmentCapacityLiters;

        Map<String, Object> data = new HashMap<>();
        data.put("binId", binId);
        data.put("date", date.toString());

        data.put("avgFillOrganic", round1(avgOrganic));
        data.put("avgFillRecycle", round1(avgRecycle));
        data.put("avgFillNonRecycle", round1(avgNonRecycle));
        data.put("avgFillHazardous", round1(avgHazardous));

        data.put("organicLiters", round1(organicLiters));
        data.put("recycleLiters", round1(recycleLiters));
        data.put("nonRecycleLiters", round1(nonRecycleLiters));
        data.put("hazardousLiters", round1(hazardousLiters));

        data.put("totalLiters", round1(
                organicLiters + recycleLiters + nonRecycleLiters + hazardousLiters
        ));

        data.put("sampleCount", sampleCount);
        data.put("compartmentCapacityLiters", compartmentCapacityLiters);
        data.put("recordedAt", Timestamp.now());

        return data;
    }

    private int safeInt(Integer value) {
        return value != null ? value : 0;
    }

    private double round1(double value) {
        return Math.round(value * 10.0) / 10.0;
    }
}
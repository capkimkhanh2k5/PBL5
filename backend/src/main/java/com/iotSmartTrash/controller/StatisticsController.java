package com.iotSmartTrash.controller;

import com.google.cloud.firestore.DocumentSnapshot;
import com.google.cloud.firestore.Firestore;
import com.google.cloud.firestore.Query;
import com.google.cloud.firestore.QuerySnapshot;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.time.LocalDate;
import java.time.ZoneId;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@RestController
@RequiredArgsConstructor
public class StatisticsController {

    private static final String DAILY_STATISTICS_COLLECTION = "bin_daily_statistics";
    private static final ZoneId ZONE = ZoneId.of("Asia/Ho_Chi_Minh");

    private static final ExecutorService FIRESTORE_EXECUTOR =
            Executors.newFixedThreadPool(2);

    private final Firestore firestore;

    /**
     * Chỉ dùng làm fallback nếu document cũ chưa có recycleLiters.
     * Bình thường API sẽ đọc recycleLiters đã tổng hợp sẵn.
     */
    @Value("${smartbin.compartment-capacity-liters}")
    private double compartmentCapacityLiters;

    @GetMapping("/api/v1/statistics/recycle/weekly")
    public Map<String, Object> getWeeklyRecycleStatistics(
            @RequestParam(required = false) String binId,
            @RequestParam(required = false) String startDate,
            @RequestParam(required = false) String endDate
    ) throws Exception {

        LocalDate end = endDate == null
                ? LocalDate.now(ZONE)
                : LocalDate.parse(endDate);

        LocalDate start = startDate == null
                ? end.minusDays(6)
                : LocalDate.parse(startDate);

        long numberOfDays = ChronoUnit.DAYS.between(start, end) + 1;

        LocalDate previousEnd = start.minusDays(1);
        LocalDate previousStart = previousEnd.minusDays(numberOfDays - 1);

        CompletableFuture<Map<LocalDate, Double>> currentFuture =
                CompletableFuture.supplyAsync(() -> {
                    try {
                        return calculateDailyRecycleLiters(binId, start, end);
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }, FIRESTORE_EXECUTOR);

        CompletableFuture<Map<LocalDate, Double>> previousFuture =
                CompletableFuture.supplyAsync(() -> {
                    try {
                        return calculateDailyRecycleLiters(binId, previousStart, previousEnd);
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }, FIRESTORE_EXECUTOR);

        Map<LocalDate, Double> currentDaily = currentFuture.get();
        Map<LocalDate, Double> previousDaily = previousFuture.get();

        double currentTotal = currentDaily.values()
                .stream()
                .mapToDouble(Double::doubleValue)
                .sum();

        double previousTotal = previousDaily.values()
                .stream()
                .mapToDouble(Double::doubleValue)
                .sum();

        int percentChange = 0;

        if (previousTotal > 0) {
            percentChange = (int) Math.round(
                    ((currentTotal - previousTotal) / previousTotal) * 100.0
            );
        }

        List<Map<String, Object>> days = new ArrayList<>();

        for (int i = 0; i < numberOfDays; i++) {
            LocalDate date = start.plusDays(i);

            Map<String, Object> day = new LinkedHashMap<>();
            day.put("date", date.toString());
            day.put("label", getDayLabel(date));
            day.put("liters", round1(currentDaily.getOrDefault(date, 0.0)));

            days.add(day);
        }

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("binId", binId == null || binId.isBlank() ? "all" : binId);
        result.put("startDate", start.toString());
        result.put("endDate", end.toString());
        result.put("totalLiters", round1(currentTotal));
        result.put("previousTotalLiters", round1(previousTotal));
        result.put("percentChange", percentChange);
        result.put("days", days);

        return result;
    }

    private Map<LocalDate, Double> calculateDailyRecycleLiters(
            String binId,
            LocalDate start,
            LocalDate end
    ) throws Exception {

        Map<LocalDate, Double> daily = new LinkedHashMap<>();

        long numberOfDays = ChronoUnit.DAYS.between(start, end) + 1;

        for (int i = 0; i < numberOfDays; i++) {
            daily.put(start.plusDays(i), 0.0);
        }

        Query query = firestore.collection(DAILY_STATISTICS_COLLECTION)
                .whereGreaterThanOrEqualTo("date", start.toString())
                .whereLessThanOrEqualTo("date", end.toString());

        if (binId != null && !binId.isBlank()) {
            query = query.whereEqualTo("binId", binId);
        }

        QuerySnapshot snapshot = query.get().get();

        for (DocumentSnapshot doc : snapshot.getDocuments()) {
            String rawDate = getString(doc, "date");

            if (rawDate == null || rawDate.isBlank()) {
                continue;
            }

            LocalDate date = LocalDate.parse(rawDate);

            if (!daily.containsKey(date)) {
                continue;
            }

            Double recycleLiters = getDouble(doc, "recycleLiters");

            /**
             * Fallback cho document cũ:
             * Nếu chưa có recycleLiters thì lấy avgFillRecycle để tính.
             */
            if (recycleLiters == null) {
                Double avgFillRecycle = getDouble(doc, "avgFillRecycle");

                if (avgFillRecycle == null) {
                    avgFillRecycle = getDouble(doc, "avg_fill_recycle");
                }

                if (avgFillRecycle == null) {
                    continue;
                }

                recycleLiters = avgFillRecycle / 100.0 * compartmentCapacityLiters;
            }

            daily.put(date, daily.get(date) + recycleLiters);
        }

        return daily;
    }

    private Double getDouble(DocumentSnapshot doc, String fieldName) {
        Object value = doc.get(fieldName);

        if (value == null) return null;

        if (value instanceof Number number) {
            return number.doubleValue();
        }

        try {
            return Double.parseDouble(value.toString());
        } catch (Exception e) {
            return null;
        }
    }

    private String getString(DocumentSnapshot doc, String fieldName) {
        Object value = doc.get(fieldName);
        return value != null ? value.toString() : null;
    }

    private String getDayLabel(LocalDate date) {
        return switch (date.getDayOfWeek()) {
            case MONDAY -> "Mon";
            case TUESDAY -> "Tue";
            case WEDNESDAY -> "Wed";
            case THURSDAY -> "Thu";
            case FRIDAY -> "Fri";
            case SATURDAY -> "Sat";
            case SUNDAY -> "Sun";
        };
    }

    private double round1(double value) {
        return Math.round(value * 10.0) / 10.0;
    }
}
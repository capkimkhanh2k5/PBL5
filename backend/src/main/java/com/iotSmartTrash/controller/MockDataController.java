package com.iotSmartTrash.controller;

import com.google.cloud.Timestamp;
import com.google.cloud.firestore.Firestore;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.HashMap;
import java.util.Map;

@RestController
@RequiredArgsConstructor
public class MockDataController {

    private final Firestore firestore;

    @GetMapping("/api/mock/bin002/raw-logs")
    public String createBin002MockRawLogs() {
        String binId = "bin_002";

        LocalDateTime startTime = LocalDateTime.of(2026, 5, 7, 0, 0);

        for (int i = 0; i < 288; i++) {
            LocalDateTime currentTime = startTime.plusMinutes(i * 5L);

            long recordedAtMillis = currentTime
                    .atZone(ZoneId.of("Asia/Ho_Chi_Minh"))
                    .toInstant()
                    .toEpochMilli();

            int minuteOfDay = currentTime.getHour() * 60 + currentTime.getMinute();

            int organic;
            int recycle;
            int nonRecycle;
            int hazardous;

            if (minuteOfDay <= 360) {
                organic = interpolate(minuteOfDay, 0, 360, 2, 8);
                recycle = interpolate(minuteOfDay, 0, 360, 1, 6);
                nonRecycle = interpolate(minuteOfDay, 0, 360, 2, 7);
                hazardous = interpolate(minuteOfDay, 0, 360, 0, 1);
            } else if (minuteOfDay <= 540) {
                organic = interpolate(minuteOfDay, 360, 540, 8, 25);
                recycle = interpolate(minuteOfDay, 360, 540, 6, 22);
                nonRecycle = interpolate(minuteOfDay, 360, 540, 7, 24);
                hazardous = interpolate(minuteOfDay, 360, 540, 1, 4);
            } else if (minuteOfDay <= 660) {
                organic = interpolate(minuteOfDay, 540, 660, 25, 40);
                recycle = interpolate(minuteOfDay, 540, 660, 22, 36);
                nonRecycle = interpolate(minuteOfDay, 540, 660, 24, 38);
                hazardous = interpolate(minuteOfDay, 540, 660, 4, 8);
            } else if (minuteOfDay <= 780) {
                organic = interpolate(minuteOfDay, 660, 780, 40, 78);
                recycle = interpolate(minuteOfDay, 660, 780, 36, 68);
                nonRecycle = interpolate(minuteOfDay, 660, 780, 38, 75);
                hazardous = interpolate(minuteOfDay, 660, 780, 8, 15);
            } else if (minuteOfDay <= 840) {
                organic = interpolate(minuteOfDay, 780, 840, 78, 95);
                recycle = interpolate(minuteOfDay, 780, 840, 68, 88);
                nonRecycle = interpolate(minuteOfDay, 780, 840, 75, 96);
                hazardous = interpolate(minuteOfDay, 780, 840, 15, 20);
            } else if (minuteOfDay == 845) {
                organic = 0;
                recycle = 0;
                nonRecycle = 0;
                hazardous = 0;
            } else if (minuteOfDay <= 1020) {
                organic = interpolate(minuteOfDay, 850, 1020, 1, 22);
                recycle = interpolate(minuteOfDay, 850, 1020, 1, 18);
                nonRecycle = interpolate(minuteOfDay, 850, 1020, 1, 24);
                hazardous = interpolate(minuteOfDay, 850, 1020, 0, 3);
            } else if (minuteOfDay <= 1200) {
                organic = interpolate(minuteOfDay, 1020, 1200, 22, 58);
                recycle = interpolate(minuteOfDay, 1020, 1200, 18, 52);
                nonRecycle = interpolate(minuteOfDay, 1020, 1200, 24, 60);
                hazardous = interpolate(minuteOfDay, 1020, 1200, 3, 8);
            } else if (minuteOfDay <= 1290) {
                organic = interpolate(minuteOfDay, 1200, 1290, 58, 82);
                recycle = interpolate(minuteOfDay, 1200, 1290, 52, 75);
                nonRecycle = interpolate(minuteOfDay, 1200, 1290, 60, 85);
                hazardous = interpolate(minuteOfDay, 1200, 1290, 8, 12);
            } else if (minuteOfDay < 1320) {
                organic = 85;
                recycle = 78;
                nonRecycle = 88;
                hazardous = 13;
            } else if (minuteOfDay == 1320) {
                organic = 0;
                recycle = 0;
                nonRecycle = 0;
                hazardous = 0;
            } else {
                organic = interpolate(minuteOfDay, 1325, 1435, 0, 8);
                recycle = interpolate(minuteOfDay, 1325, 1435, 0, 6);
                nonRecycle = interpolate(minuteOfDay, 1325, 1435, 0, 9);
                hazardous = interpolate(minuteOfDay, 1325, 1435, 0, 1);
            }

            organic = clamp(organic);
            recycle = clamp(recycle);
            nonRecycle = clamp(nonRecycle);
            hazardous = clamp(hazardous);

            Map<String, Object> log = new HashMap<>();
            log.put("fillOrganic", organic * 1.0);
            log.put("fillRecycle", recycle * 1.0);
            log.put("fillNonRecycle", nonRecycle * 1.0);
            log.put("fillHazardous", hazardous * 1.0);
            log.put(
                    "recordedAt",
                    Timestamp.ofTimeSecondsAndNanos(recordedAtMillis / 1000, 0)
            );

            firestore.collection("bin_raw_sensor_logs")
                    .document(binId)
                    .collection("logs")
                    .add(log);
        }

        return "Created one-day mock raw sensor logs for bin_002 successfully";
    }

    private int interpolate(int current, int start, int end, int startValue, int endValue) {
        if (current <= start) return startValue;
        if (current >= end) return endValue;

        double ratio = (double) (current - start) / (end - start);
        return (int) Math.round(startValue + ratio * (endValue - startValue));
    }

    private int clamp(int value) {
        return Math.max(0, Math.min(100, value));
    }
}
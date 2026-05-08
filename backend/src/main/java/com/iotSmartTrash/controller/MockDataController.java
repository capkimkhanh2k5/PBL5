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
            int minuteOfDay = currentTime.getHour() * 60 + currentTime.getMinute();

            double nonRecycle = getNonRecycle(minuteOfDay);
            double organic = getOrganic(minuteOfDay);
            double recycle = getRecycle(minuteOfDay);
            double hazardous = getHazardous(minuteOfDay);

            long recordedAtMillis = currentTime
                    .atZone(ZoneId.of("Asia/Ho_Chi_Minh"))
                    .toInstant()
                    .toEpochMilli();

            Map<String, Object> log = new HashMap<>();
            log.put("fillNonRecycle", round1(nonRecycle));
            log.put("fillOrganic", round1(organic));
            log.put("fillRecycle", round1(recycle));
            log.put("fillHazardous", round1(hazardous));
            log.put("recordedAt", Timestamp.ofTimeSecondsAndNanos(recordedAtMillis / 1000, 0));

            firestore.collection("bin_raw_sensor_logs")
                    .document(binId)
                    .collection("logs")
                    .add(log);
        }

        return "Created fixed mock data like demo chart.";
    }

    private double getNonRecycle(int m) {
        return interpolate(m, new int[]{0, 180, 360, 420, 480, 525, 535, 540, 600, 720, 840, 960, 1080, 1200, 1320, 1435},
                new double[]{65, 66.5, 68, 69, 73, 86, 91, 3, 4, 10, 15, 18, 30, 38, 41, 43});
    }

    private double getOrganic(int m) {
        return interpolate(m, new int[]{0, 180, 360, 480, 525, 535, 540, 660, 720, 780, 900, 1020, 1080, 1200, 1320, 1435},
                new double[]{40, 40.5, 41, 43, 44, 45, 2, 5, 12, 18, 21, 23, 28, 32, 34, 35});
    }

    private double getRecycle(int m) {
        return interpolate(m, new int[]{0, 180, 360, 480, 525, 535, 540, 660, 780, 900, 960, 1080, 1200, 1320, 1435},
                new double[]{49, 49.5, 50, 53, 55, 57, 1.5, 4, 10, 18, 25, 33, 36, 38, 39});
    }

    private double getHazardous(int m) {
        return interpolate(m, new int[]{0, 360, 480, 535, 540, 720, 840, 960, 1200, 1435},
                new double[]{5, 5.5, 6.0, 6.8, 1.3, 1.7, 3.2, 3.5, 4.4, 5.0});
    }

    private double interpolate(int minute, int[] times, double[] values) {
        if (minute <= times[0]) return values[0];

        for (int i = 0; i < times.length - 1; i++) {
            if (minute >= times[i] && minute <= times[i + 1]) {
                double ratio = (minute - times[i]) * 1.0 / (times[i + 1] - times[i]);
                return values[i] + ratio * (values[i + 1] - values[i]);
            }
        }

        return values[values.length - 1];
    }

    private double round1(double value) {
        return Math.round(value * 10.0) / 10.0;
    }
}
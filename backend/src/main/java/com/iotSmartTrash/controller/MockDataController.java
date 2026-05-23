package com.iotSmartTrash.controller;

import com.google.cloud.Timestamp;
import com.google.cloud.firestore.CollectionReference;
import com.google.cloud.firestore.Firestore;
import com.google.cloud.firestore.QueryDocumentSnapshot;
import com.google.cloud.firestore.WriteBatch;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequiredArgsConstructor
public class MockDataController {

    private final Firestore firestore;

    private static final ZoneId VN_ZONE = ZoneId.of("Asia/Ho_Chi_Minh");

    private static final DateTimeFormatter DOC_ID_FORMAT =
            DateTimeFormatter.ofPattern("yyyyMMdd_HHmm");

    @GetMapping("/api/mock/bin003/reset-14-days")
    public String resetBin003MockData14Days() throws Exception {
        String binId = "bin_003";

        System.out.println("Start reset mock data for " + binId);

        int deletedRawLogs = clearCollection(
                firestore.collection("bin_raw_sensor_logs")
                        .document(binId)
                        .collection("logs")
        );

        int deletedEvents = clearCollection(
                firestore.collection("bin_collection_events")
                        .document(binId)
                        .collection("events")
        );

        ensureBinParentDocuments(binId);

        MockResult result = createMockData14Days(binId);

        return "Reset mock data for " + binId
                + ". Deleted raw logs: " + deletedRawLogs
                + ", deleted events: " + deletedEvents
                + ", created raw logs: " + result.rawLogCount
                + ", created events: " + result.eventCount;
    }

    private void ensureBinParentDocuments(String binId) throws Exception {
        Map<String, Object> rawParent = new HashMap<>();
        rawParent.put("binId", binId);
        rawParent.put("type", "raw_sensor_logs");
        rawParent.put("updatedAt", Timestamp.now());

        firestore.collection("bin_raw_sensor_logs")
                .document(binId)
                .set(rawParent)
                .get();

        Map<String, Object> eventParent = new HashMap<>();
        eventParent.put("binId", binId);
        eventParent.put("type", "collection_events");
        eventParent.put("updatedAt", Timestamp.now());

        firestore.collection("bin_collection_events")
                .document(binId)
                .set(eventParent)
                .get();
    }

    private int clearCollection(CollectionReference collection) throws Exception {
        int totalDeleted = 0;

        while (true) {
            List<QueryDocumentSnapshot> docs =
                    collection.limit(450).get().get().getDocuments();

            if (docs.isEmpty()) {
                break;
            }

            WriteBatch batch = firestore.batch();

            for (QueryDocumentSnapshot doc : docs) {
                batch.delete(doc.getReference());
            }

            batch.commit().get();

            totalDeleted += docs.size();

            System.out.println("Deleted documents: " + totalDeleted);
        }

        return totalDeleted;
    }

    private MockResult createMockData14Days(String binId) throws Exception {
        LocalDate today = LocalDate.now(VN_ZONE);
        LocalDate startDate = today.minusDays(13);

        int rawLogCount = 0;
        int eventCount = 0;
        int batchOps = 0;

        WriteBatch batch = firestore.batch();

        for (int day = 0; day < 14; day++) {
            LocalDate currentDate = startDate.plusDays(day);
            LocalDateTime dayStart = currentDate.atStartOfDay();

            DayScenario scenario = getScenarioForDay(day);

            double nonRecycle = scenario.startNonRecycle;
            double organic = scenario.startOrganic;
            double recycle = scenario.startRecycle;
            double hazardous = scenario.startHazardous;

            for (int i = 0; i < 288; i++) {
                LocalDateTime currentTime = dayStart.plusMinutes(i * 5L);
                int minuteOfDay = currentTime.getHour() * 60 + currentTime.getMinute();

                boolean isEmptyTime = containsMinute(scenario.emptyTimes, minuteOfDay);

                if (isEmptyTime) {
                    double beforeRecycle = recycle;

                    nonRecycle = scenario.afterEmptyNonRecycle + smallNoise(day, minuteOfDay, 1, 0.8);
                    organic = scenario.afterEmptyOrganic + smallNoise(day, minuteOfDay, 2, 0.7);
                    recycle = scenario.afterEmptyRecycle + smallNoise(day, minuteOfDay, 3, 0.6);
                    hazardous = scenario.afterEmptyHazardous + smallNoise(day, minuteOfDay, 4, 0.2);

                    nonRecycle = clamp(nonRecycle, 0, 96);
                    organic = clamp(organic, 0, 96);
                    recycle = clamp(recycle, 0, 96);
                    hazardous = clamp(hazardous, 0, 96);

                    double estimatedKg = calculateRecycleKg(
                            beforeRecycle,
                            recycle,
                            scenario.recycleCapacityKg,
                            day,
                            minuteOfDay
                    );

                    Map<String, Object> event = new HashMap<>();
                    event.put("binId", binId);
                    event.put("wasteType", "recycle");
                    event.put("estimatedKg", round1(estimatedKg));
                    event.put("beforeFillPercent", round1(beforeRecycle));
                    event.put("afterFillPercent", round1(recycle));
                    event.put("emptiedAt", toTimestamp(currentTime));
                    event.put("date", currentDate.toString());
                    event.put("scenario", scenario.name);

                    String eventId = currentTime.format(DOC_ID_FORMAT) + "_recycle";

                    batch.set(
                            firestore.collection("bin_collection_events")
                                    .document(binId)
                                    .collection("events")
                                    .document(eventId),
                            event
                    );

                    batchOps++;
                    eventCount++;
                } else {
                    nonRecycle += getNonRecycleIncrease(minuteOfDay, day, scenario);
                    organic += getOrganicIncrease(minuteOfDay, day, scenario);
                    recycle += getRecycleIncrease(minuteOfDay, day, scenario);
                    hazardous += getHazardousIncrease(minuteOfDay, day, scenario);

                    nonRecycle = clamp(nonRecycle, 0, 96);
                    organic = clamp(organic, 0, 96);
                    recycle = clamp(recycle, 0, 96);
                    hazardous = clamp(hazardous, 0, 96);
                }

                Map<String, Object> log = new HashMap<>();
                log.put("binId", binId);
                log.put("fillNonRecycle", round1(nonRecycle));
                log.put("fillOrganic", round1(organic));
                log.put("fillRecycle", round1(recycle));
                log.put("fillHazardous", round1(hazardous));
                log.put("recordedAt", toTimestamp(currentTime));
                log.put("scenario", scenario.name);

                String documentId = currentTime.format(DOC_ID_FORMAT);

                batch.set(
                        firestore.collection("bin_raw_sensor_logs")
                                .document(binId)
                                .collection("logs")
                                .document(documentId),
                        log
                );

                batchOps++;
                rawLogCount++;

                if (batchOps >= 450) {
                    batch.commit().get();

                    System.out.println("Committed raw logs: " + rawLogCount
                            + ", events: " + eventCount);

                    batch = firestore.batch();
                    batchOps = 0;
                }
            }
        }

        if (batchOps > 0) {
            batch.commit().get();
        }

        System.out.println("Finished raw logs: " + rawLogCount);
        System.out.println("Finished events: " + eventCount);

        return new MockResult(rawLogCount, eventCount);
    }

    private DayScenario getScenarioForDay(int day) {
        DayScenario[] scenarios = new DayScenario[]{
                new DayScenario(
                        "Quiet Monday - light usage",
                        new int[]{8 * 60 + 45},
                        31, 22, 28, 3.5,
                        3.5, 2.8, 2.5, 0.9,
                        0.85,
                        24.0
                ),
                new DayScenario(
                        "Busy Tuesday - morning and evening collection",
                        new int[]{9 * 60 + 20, 17 * 60 + 10},
                        38, 29, 34, 4.2,
                        4.0, 3.2, 2.8, 1.0,
                        1.18,
                        25.5
                ),
                new DayScenario(
                        "Normal Wednesday - lunch peak",
                        new int[]{11 * 60 + 30},
                        35, 31, 30, 3.8,
                        3.8, 3.0, 2.6, 0.8,
                        1.05,
                        24.5
                ),
                new DayScenario(
                        "Student activity day",
                        new int[]{10 * 60 + 15},
                        42, 26, 37, 4.5,
                        4.5, 3.5, 3.0, 1.1,
                        1.22,
                        26.0
                ),
                new DayScenario(
                        "Event day - two collections",
                        new int[]{8 * 60 + 50, 16 * 60 + 40},
                        46, 35, 41, 5.0,
                        4.8, 3.8, 3.2, 1.2,
                        1.35,
                        27.0
                ),
                new DayScenario(
                        "Rainy day - fewer users",
                        new int[]{13 * 60 + 25},
                        28, 20, 25, 3.1,
                        3.3, 2.6, 2.2, 0.8,
                        0.72,
                        23.5
                ),
                new DayScenario(
                        "Weekend light traffic",
                        new int[]{15 * 60 + 10},
                        24, 18, 22, 2.8,
                        3.0, 2.4, 2.0, 0.7,
                        0.65,
                        23.0
                ),
                new DayScenario(
                        "High traffic morning and evening",
                        new int[]{9 * 60 + 5, 18 * 60},
                        43, 33, 39, 4.7,
                        4.2, 3.4, 3.0, 1.0,
                        1.28,
                        26.5
                ),
                new DayScenario(
                        "Office day - stable usage",
                        new int[]{12 * 60 + 20},
                        36, 27, 32, 3.9,
                        3.9, 3.1, 2.7, 0.9,
                        1.00,
                        24.8
                ),
                new DayScenario(
                        "Library peak day",
                        new int[]{10 * 60 + 45},
                        40, 24, 38, 4.0,
                        4.1, 3.0, 2.9, 0.9,
                        1.16,
                        25.8
                ),
                new DayScenario(
                        "Canteen busy day - two collections",
                        new int[]{8 * 60 + 35, 17 * 60 + 30},
                        45, 38, 36, 4.9,
                        4.7, 3.9, 3.1, 1.1,
                        1.32,
                        27.2
                ),
                new DayScenario(
                        "Afternoon peak",
                        new int[]{14 * 60 + 15},
                        34, 28, 33, 3.6,
                        3.7, 2.9, 2.5, 0.8,
                        1.07,
                        24.6
                ),
                new DayScenario(
                        "Moderate traffic day",
                        new int[]{11 * 60 + 10},
                        37, 25, 31, 3.7,
                        3.6, 2.8, 2.4, 0.8,
                        0.96,
                        24.3
                ),
                new DayScenario(
                        "Final demo day - high recycle usage",
                        new int[]{9 * 60 + 40, 16 * 60 + 55},
                        41, 30, 44, 4.4,
                        4.3, 3.2, 3.0, 1.0,
                        1.25,
                        26.8
                )
        };

        return scenarios[day % scenarios.length];
    }

    private double getNonRecycleIncrease(int minute, int day, DayScenario scenario) {
        double value = 0.045 * scenario.trafficFactor;

        if (isBetween(minute, 7, 9)) value += 0.045;
        if (isBetween(minute, 11, 13)) value += 0.060;
        if (isBetween(minute, 16, 19)) value += 0.055;

        value += smallNoise(day, minute, 11, 0.018);

        return Math.max(value, 0.005);
    }

    private double getOrganicIncrease(int minute, int day, DayScenario scenario) {
        double value = 0.032 * scenario.trafficFactor;

        if (isBetween(minute, 6, 8)) value += 0.025;
        if (isBetween(minute, 11, 13)) value += 0.085;
        if (isBetween(minute, 17, 19)) value += 0.045;

        value += smallNoise(day, minute, 22, 0.015);

        return Math.max(value, 0.004);
    }

    private double getRecycleIncrease(int minute, int day, DayScenario scenario) {
        double value = 0.040 * scenario.trafficFactor;

        if (isBetween(minute, 8, 10)) value += 0.045;
        if (isBetween(minute, 12, 14)) value += 0.055;
        if (isBetween(minute, 15, 18)) value += 0.075;
        if (isBetween(minute, 19, 21)) value += 0.030;

        value += smallNoise(day, minute, 33, 0.020);

        return Math.max(value, 0.005);
    }

    private double getHazardousIncrease(int minute, int day, DayScenario scenario) {
        double value = 0.004 * scenario.trafficFactor;

        if (isBetween(minute, 9, 11)) value += 0.004;
        if (isBetween(minute, 14, 17)) value += 0.006;

        value += smallNoise(day, minute, 44, 0.003);

        return Math.max(value, 0.0005);
    }

    private boolean isBetween(int minuteOfDay, int startHour, int endHour) {
        return minuteOfDay >= startHour * 60 && minuteOfDay < endHour * 60;
    }

    private boolean containsMinute(int[] times, int minute) {
        for (int time : times) {
            if (time == minute) {
                return true;
            }
        }

        return false;
    }

    private double calculateRecycleKg(
            double beforeFill,
            double afterFill,
            double capacityKg,
            int day,
            int minute
    ) {
        double collectedPercent = Math.max(0, beforeFill - afterFill);

        double densityFactor = 0.92
                + Math.abs(Math.sin(day * 0.77 + minute * 0.013)) * 0.16;

        return collectedPercent / 100.0 * capacityKg * densityFactor;
    }

    private double smallNoise(int day, int minute, int salt, double amplitude) {
        return Math.sin(day * 12.9898 + minute * 0.017 + salt * 7.131) * amplitude;
    }

    private Timestamp toTimestamp(LocalDateTime time) {
        long millis = time
                .atZone(VN_ZONE)
                .toInstant()
                .toEpochMilli();

        return Timestamp.ofTimeSecondsAndNanos(millis / 1000, 0);
    }

    private double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    private double round1(double value) {
        return Math.round(value * 10.0) / 10.0;
    }

    private static class MockResult {
        final int rawLogCount;
        final int eventCount;

        MockResult(int rawLogCount, int eventCount) {
            this.rawLogCount = rawLogCount;
            this.eventCount = eventCount;
        }
    }

    private static class DayScenario {
        final String name;
        final int[] emptyTimes;

        final double startNonRecycle;
        final double startOrganic;
        final double startRecycle;
        final double startHazardous;

        final double afterEmptyNonRecycle;
        final double afterEmptyOrganic;
        final double afterEmptyRecycle;
        final double afterEmptyHazardous;

        final double trafficFactor;
        final double recycleCapacityKg;

        DayScenario(
                String name,
                int[] emptyTimes,
                double startNonRecycle,
                double startOrganic,
                double startRecycle,
                double startHazardous,
                double afterEmptyNonRecycle,
                double afterEmptyOrganic,
                double afterEmptyRecycle,
                double afterEmptyHazardous,
                double trafficFactor,
                double recycleCapacityKg
        ) {
            this.name = name;
            this.emptyTimes = emptyTimes;
            this.startNonRecycle = startNonRecycle;
            this.startOrganic = startOrganic;
            this.startRecycle = startRecycle;
            this.startHazardous = startHazardous;
            this.afterEmptyNonRecycle = afterEmptyNonRecycle;
            this.afterEmptyOrganic = afterEmptyOrganic;
            this.afterEmptyRecycle = afterEmptyRecycle;
            this.afterEmptyHazardous = afterEmptyHazardous;
            this.trafficFactor = trafficFactor;
            this.recycleCapacityKg = recycleCapacityKg;
        }
    }
}
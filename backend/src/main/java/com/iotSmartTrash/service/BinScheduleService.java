package com.iotSmartTrash.service;

import com.google.cloud.firestore.DocumentSnapshot;
import com.google.cloud.firestore.Firestore;
import com.google.cloud.firestore.Query;
import com.google.cloud.firestore.QueryDocumentSnapshot;
import com.iotSmartTrash.dto.BinPickupScheduleDTO;
import com.iotSmartTrash.exception.ServiceException;
import com.iotSmartTrash.model.BinSensorLog;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

@Service
@RequiredArgsConstructor
public class BinScheduleService {

    private static final String COLLECTION_NAME = "bin_sensor_logs";

    private final Firestore firestore;

    public List<BinPickupScheduleDTO> getPickupSchedule(int limit) {
        try {
            int expandedLimit = Math.max(1, Math.min(limit, 100)) * 4;
            Query query = firestore.collection(COLLECTION_NAME)
                    .orderBy("recorded_at", Query.Direction.DESCENDING)
                    .limit(expandedLimit);

            Map<String, BinPickupScheduleDTO> latestByBin = new HashMap<>();
            for (QueryDocumentSnapshot doc : query.get().get().getDocuments()) {
                BinSensorLog log = doc.toObject(BinSensorLog.class);
                if (log == null || log.getBinId() == null || log.getBinId().isBlank()) {
                    continue;
                }

                int avgFill = averageFill(log);
                int fillOrganic = safeInt(log.getAvgFillOrganic());
                int fillRecycle = safeInt(log.getAvgFillRecycle());
                int fillNonRecycle = safeInt(log.getAvgFillNonRecycle());
                int fillHazardous = safeInt(log.getAvgFillHazardous());

                double urgencyScore = calculateUrgencyScore(fillOrganic, fillRecycle, fillNonRecycle, fillHazardous);
                int etaHours = estimatePickupInHours(urgencyScore, fillHazardous);
                long sourceRecordedAt = extractRecordedAtMillis(doc, log);
                long baseTime = sourceRecordedAt > 0 ? sourceRecordedAt : Instant.now().toEpochMilli();
                long predictedPickupAt = baseTime + etaHours * 3600_000L;

                BinPickupScheduleDTO item = BinPickupScheduleDTO.builder()
                        .binId(log.getBinId())
                        .avgFill(avgFill)
                        .sourceDate(log.getDate())
                        .sourcePeriod(log.getPeriod() != null ? log.getPeriod().toLabel() : null)
                        .sourceRecordedAt(sourceRecordedAt)
                        .predictedPickupAt(predictedPickupAt)
                        .predictedInHours(etaHours)
                        .priority(priorityByUrgency(urgencyScore, fillHazardous))
                        .build();

                BinPickupScheduleDTO current = latestByBin.get(log.getBinId());
                if (current == null || safeLong(current.getSourceRecordedAt()) < sourceRecordedAt) {
                    latestByBin.put(log.getBinId(), item);
                }
            }

            List<BinPickupScheduleDTO> items = new ArrayList<>(latestByBin.values());
            items.sort(Comparator.comparing(BinPickupScheduleDTO::getPredictedPickupAt));
            int safeLimit = Math.max(1, Math.min(limit, 100));
            return items.size() <= safeLimit ? items : items.subList(0, safeLimit);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot build pickup schedule: operation interrupted", e);
        } catch (ExecutionException e) {
            throw new ServiceException("Cannot build pickup schedule", e.getCause());
        }
    }

    private static int averageFill(BinSensorLog log) {
        int org = safeInt(log.getAvgFillOrganic());
        int rec = safeInt(log.getAvgFillRecycle());
        int non = safeInt(log.getAvgFillNonRecycle());
        int haz = safeInt(log.getAvgFillHazardous());
        return (org + rec + non + haz) / 4;
    }

    private static double calculateUrgencyScore(int organic, int recycle, int nonRecycle, int hazardous) {
        // Heavier weight for non-recycle and hazardous compartments.
        return (organic * 0.20) + (recycle * 0.15) + (nonRecycle * 0.30) + (hazardous * 0.35);
    }

    private static int estimatePickupInHours(double urgencyScore, int hazardousFill) {
        if (hazardousFill >= 90) return 4;
        if (urgencyScore >= 90) return 6;
        if (urgencyScore >= 80) return 10;
        if (urgencyScore >= 70) return 16;
        if (urgencyScore >= 55) return 24;
        if (urgencyScore >= 40) return 36;
        return 48;
    }

    private static String priorityByUrgency(double urgencyScore, int hazardousFill) {
        if (hazardousFill >= 90 || urgencyScore >= 90) return "CRITICAL";
        if (hazardousFill >= 80 || urgencyScore >= 80) return "HIGH";
        if (urgencyScore >= 65) return "MEDIUM";
        return "LOW";
    }

    private static int safeInt(Integer value) {
        return value != null ? value : 0;
    }

    private static long safeLong(Long value) {
        return value != null ? value : 0L;
    }

    private static long extractRecordedAtMillis(DocumentSnapshot doc, BinSensorLog log) {
        if (log.getRecordedAt() != null) {
            return log.getRecordedAt().toDate().getTime();
        }

        Object raw = doc.get("recorded_at");
        if (raw instanceof com.google.cloud.Timestamp ts) {
            return ts.toDate().getTime();
        }
        return 0L;
    }
}

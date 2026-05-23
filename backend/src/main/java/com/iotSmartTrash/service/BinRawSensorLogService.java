package com.iotSmartTrash.service;

import com.google.cloud.firestore.*;
import com.iotSmartTrash.exception.ServiceException;
import com.iotSmartTrash.model.BinRawSensorLog;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import com.google.cloud.firestore.Query;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import com.google.cloud.Timestamp;


/**
 * Service quản lý raw sensor logs trên Firestore (subcollection pattern).
 * Collection: bin_raw_sensor_logs/{bin_id}/logs/{auto_id}
 * Dữ liệu chỉ giữ 24h gần nhất — scheduler dọn dẹp mỗi 6 tiếng.
 */
@Service
@RequiredArgsConstructor
public class BinRawSensorLogService {

    private static final String PARENT_COLLECTION = "bin_raw_sensor_logs";
    private static final String SUB_COLLECTION = "logs";

    private final Firestore firestore;

    /**
     * Raspi gọi mỗi 30 giây để ghi một raw sensor log mới.
     */
    public String addLog(String binId, BinRawSensorLog log) {

        try {
            DocumentReference docRef = firestore
                    .collection(PARENT_COLLECTION).document(binId)
                    .collection(SUB_COLLECTION).document();
            long recordedAt = log.getRecordedAt() != null
                    ? log.getRecordedAt().toDate().getTime()
                    : System.currentTimeMillis();
            // Write canonical snake_case fields for stable querying/indexing.
                Map<String, Object> payload = new HashMap<>();

                payload.put("fillOrganic", safeInt(log.getFillOrganic()));
                payload.put("fillRecycle", safeInt(log.getFillRecycle()));
                payload.put("fillNonRecycle", safeInt(log.getFillNonRecycle()));
                payload.put("fillHazardous", safeInt(log.getFillHazardous()));
                payload.put("recordedAt", com.google.cloud.Timestamp.ofTimeSecondsAndNanos(
                    recordedAt / 1000,
                    (int) ((recordedAt % 1000) * 1_000_000)
                ));


            String updateTime = docRef.set(payload).get().getUpdateTime().toString();


            return updateTime;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot add raw sensor log: operation interrupted", e);
        } catch (ExecutionException e) {
            throw new ServiceException("Cannot add raw sensor log for bin: " + binId, e.getCause());
        }
    }

    /**
     * Lấy tất cả raw sensor logs của 1 thùng rác (dùng cho scheduler aggregate).
     */
    public List<BinRawSensorLog> getLogsForBin(String binId) {
        try {
            List<QueryDocumentSnapshot> docs = queryByRecordedField(
                    binId,
                    Query.Direction.ASCENDING,
                    null);

            List<BinRawSensorLog> logs = new ArrayList<>();
            for (QueryDocumentSnapshot doc : docs) {
                logs.add(mapRawLog(doc));
            }
            return logs;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot get raw sensor logs: operation interrupted", e);
        } catch (ExecutionException e) {
            throw new ServiceException("Cannot get raw sensor logs for bin: " + binId, e.getCause());
        }
    }

    /**
     * Lấy N raw sensor logs gần nhất của 1 thùng rác.
     */
    public List<BinRawSensorLog> getRecentLogsForBin(String binId, int limit) {
        try {
            List<QueryDocumentSnapshot> docs = queryByRecordedField(
                    binId,
                    Query.Direction.DESCENDING,
                    limit);

            List<BinRawSensorLog> logs = new ArrayList<>();
            for (QueryDocumentSnapshot doc : docs) {
                logs.add(mapRawLog(doc));
            }
            return logs;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot get recent raw sensor logs: operation interrupted", e);
        } catch (ExecutionException e) {
            throw new ServiceException("Cannot get recent raw sensor logs for bin: " + binId, e.getCause());
        }
    }

    /**
     * Lấy danh sách tất cả bin_id có raw sensor logs (dùng cho scheduler).
     */
    public List<String> getAllBinIds() {
        try {
            List<String> binIds = new ArrayList<>();
            for (DocumentReference docRef : firestore.collection(PARENT_COLLECTION).listDocuments()) {
                binIds.add(docRef.getId());
            }
            return binIds;
        } catch (Exception e) {
            throw new ServiceException("Cannot list bin IDs from raw sensor logs", e);
        }
    }

    /**
     * Xóa raw sensor logs cũ hơn cutoff timestamp cho 1 thùng rác.
     * Trả về số lượng docs đã xóa.
     */
    public int deleteOldLogs(String binId, long cutoffMillis) {
        try {
            int deletedCount = 0;
            CollectionReference logsRef = firestore
                    .collection(PARENT_COLLECTION).document(binId)
                    .collection(SUB_COLLECTION);

            // Query logs cũ hơn cutoff
            Set<QueryDocumentSnapshot> oldDocs = new HashSet<>();
            oldDocs.addAll(
                    logsRef.whereLessThan(
                            "recordedAt",
                            com.google.cloud.Timestamp.ofTimeSecondsAndNanos(
                                    cutoffMillis / 1000,
                                    (int)((cutoffMillis % 1000) * 1_000_000)
                            )
                    ).get().get().getDocuments()
            );
            // Xóa theo batch (tối đa 500 writes/batch theo giới hạn Firestore)
            WriteBatch batch = firestore.batch();
            int batchCount = 0;

            for (QueryDocumentSnapshot doc : oldDocs) {
                batch.delete(doc.getReference());
                batchCount++;
                deletedCount++;

                if (batchCount >= 450) { // commit trước khi chạm giới hạn 500
                    batch.commit().get();
                    batch = firestore.batch();
                    batchCount = 0;
                }
            }

            if (batchCount > 0) {
                batch.commit().get();
            }

            return deletedCount;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot delete old logs: operation interrupted", e);
        } catch (ExecutionException e) {
            throw new ServiceException("Cannot delete old logs for bin: " + binId, e.getCause());
        }
    }

    private List<QueryDocumentSnapshot> queryByRecordedField(
            String binId,
            Query.Direction direction,
            Integer limit) throws InterruptedException, ExecutionException {

        CollectionReference ref = firestore
                .collection(PARENT_COLLECTION).document(binId)
                .collection(SUB_COLLECTION);

        return buildQuery(ref, "recordedAt", direction, limit).get().get().getDocuments();
    }

    private Query buildQuery(CollectionReference ref, String orderField, Query.Direction direction, Integer limit) {
        Query query = ref.orderBy(orderField, direction);
        if (limit != null) {
            query = query.limit(limit);
        }
        return query;
    }

    private BinRawSensorLog mapRawLog(QueryDocumentSnapshot doc) {
        return BinRawSensorLog.builder()
                .id(doc.getId())

                .fillOrganic(getInt(doc, "fillOrganic"))
                .fillRecycle(getInt(doc, "fillRecycle"))
                .fillNonRecycle(getInt(doc, "fillNonRecycle"))
                .fillHazardous(getInt(doc, "fillHazardous"))
                .recordedAt(getTimestamp(doc, "recordedAt"))
                .build();
    }

    private Integer safeInt(Integer value) {
        return value != null ? value : 0;
    }

    private com.google.cloud.Timestamp getTimestamp(QueryDocumentSnapshot doc, String key) {
        Object value = doc.get(key);
        if (value instanceof com.google.cloud.Timestamp ts) {
            return ts;
        }
        if (value instanceof Long millis) {
            return com.google.cloud.Timestamp.ofTimeSecondsAndNanos(
                    millis / 1000,
                    (int)((millis % 1000) * 1_000_000)
            );
        }
        return null;
    }

    private Integer getInt(QueryDocumentSnapshot doc, String... keys) {
        for (String key : keys) {
            Long value = doc.getLong(key);
            if (value != null) {
                return value.intValue();
            }
        }
        return 0;
    }

    private Long getLong(QueryDocumentSnapshot doc, String... keys) {
        for (String key : keys) {
            Long value = doc.getLong(key);
            if (value != null) {
                return value;
            }
        }
        return 0L;
    }

    public BinRawSensorLog getLatestRawLogByBinId(String binId) {
        try {
            var snapshot = firestore.collection("bin_raw_sensor_logs")
                    .document(binId)
                    .collection("logs")
                    .orderBy("recordedAt", Query.Direction.DESCENDING)
                    .limit(1)
                    .get()
                    .get();

            if (snapshot.isEmpty()) {
                throw new ServiceException("No raw sensor logs found for bin: " + binId);
            }

            var doc = snapshot.getDocuments().get(0);
            return mapRawLog(doc);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot get latest raw log: operation interrupted", e);
        } catch (Exception e) {
            throw new ServiceException("Cannot get latest raw log for bin: " + binId, e);
        }
    }

    public List<BinRawSensorLog> getLogsForBinBetween(
            String binId,
            Timestamp start,
            Timestamp end
    ) {
        try {
            if (binId == null || binId.isBlank()) {
                return new ArrayList<>();
            }

            CollectionReference ref = firestore
                    .collection(PARENT_COLLECTION)
                    .document(binId)
                    .collection(SUB_COLLECTION);

            QuerySnapshot snapshot = ref
                    .whereGreaterThanOrEqualTo("recordedAt", start)
                    .whereLessThan("recordedAt", end)
                    .orderBy("recordedAt", Query.Direction.ASCENDING)
                    .get()
                    .get();

            List<BinRawSensorLog> logs = new ArrayList<>();

            for (QueryDocumentSnapshot doc : snapshot.getDocuments()) {
                logs.add(mapRawLog(doc));
            }

            return logs;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot get raw logs between dates: operation interrupted", e);
        } catch (Exception e) {
            throw new ServiceException("Cannot get raw logs between dates for bin: " + binId, e);
        }
    }

}


package com.iotSmartTrash.service;

import com.google.cloud.firestore.*;
import com.iotSmartTrash.exception.ServiceException;
import com.iotSmartTrash.model.BinRawSensorLog;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

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
            log.setId(docRef.getId());
            if (log.getRecordedAt() == null) {
                log.setRecordedAt(System.currentTimeMillis());
            }
            return docRef.set(log).get().getUpdateTime().toString();
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
            List<BinRawSensorLog> logs = new ArrayList<>();
            for (QueryDocumentSnapshot doc : firestore
                    .collection(PARENT_COLLECTION).document(binId)
                    .collection(SUB_COLLECTION)
                    .orderBy("recorded_at", Query.Direction.ASCENDING)
                    .get().get().getDocuments()) {
                BinRawSensorLog log = doc.toObject(BinRawSensorLog.class);
                if (log != null) {
                    log.setId(doc.getId());
                    logs.add(log);
                }
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
            List<QueryDocumentSnapshot> oldDocs = logsRef
                    .whereLessThan("recorded_at", cutoffMillis)
                    .get().get().getDocuments();

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
}

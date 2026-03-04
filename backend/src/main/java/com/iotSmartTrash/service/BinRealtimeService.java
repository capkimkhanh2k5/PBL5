package com.iotSmartTrash.service;

import com.google.cloud.firestore.DocumentReference;
import com.google.cloud.firestore.DocumentSnapshot;
import com.google.cloud.firestore.Firestore;
import com.google.cloud.firestore.QueryDocumentSnapshot;
import com.iotSmartTrash.exception.ResourceNotFoundException;
import com.iotSmartTrash.exception.ServiceException;
import com.iotSmartTrash.model.BinRealtimeStatus;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

/**
 * Service quản lý trạng thái realtime của thùng rác trên Firestore.
 * Collection: bin_realtime_status/{bin_id}
 */
@Service
@RequiredArgsConstructor
public class BinRealtimeService {

    private static final String COLLECTION_NAME = "bin_realtime_status";

    private final Firestore firestore;

    /**
     * Raspi gọi mỗi 30 giây để cập nhật trạng thái hiện tại của thùng rác.
     * Sử dụng set() để upsert (ghi đè nếu đã tồn tại).
     */
    public String updateStatus(String binId, BinRealtimeStatus status) {
        try {
            status.setId(binId);
            status.setLastUpdated(System.currentTimeMillis());
            return firestore.collection(COLLECTION_NAME).document(binId)
                    .set(status).get().getUpdateTime().toString();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot update bin realtime status: operation interrupted", e);
        } catch (ExecutionException e) {
            throw new ServiceException("Cannot update bin realtime status: " + binId, e.getCause());
        }
    }

    /**
     * Lấy trạng thái realtime của 1 thùng rác.
     */
    public BinRealtimeStatus getStatus(String binId) {
        try {
            DocumentSnapshot doc = firestore.collection(COLLECTION_NAME).document(binId).get().get();
            if (!doc.exists()) {
                throw new ResourceNotFoundException("Bin realtime status", binId);
            }
            BinRealtimeStatus status = doc.toObject(BinRealtimeStatus.class);
            if (status != null) {
                status.setId(doc.getId());
            }
            return status;
        } catch (ResourceNotFoundException e) {
            throw e;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot get bin realtime status: operation interrupted", e);
        } catch (ExecutionException e) {
            throw new ServiceException("Cannot get bin realtime status: " + binId, e.getCause());
        }
    }

    /**
     * Lấy trạng thái realtime của tất cả thùng rác.
     */
    public List<BinRealtimeStatus> getAllStatuses() {
        try {
            List<BinRealtimeStatus> statuses = new ArrayList<>();
            for (QueryDocumentSnapshot doc : firestore.collection(COLLECTION_NAME).get().get().getDocuments()) {
                BinRealtimeStatus status = doc.toObject(BinRealtimeStatus.class);
                if (status != null) {
                    status.setId(doc.getId());
                    statuses.add(status);
                }
            }
            return statuses;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot get all bin realtime statuses: operation interrupted", e);
        } catch (ExecutionException e) {
            throw new ServiceException("Cannot get all bin realtime statuses", e.getCause());
        }
    }
}

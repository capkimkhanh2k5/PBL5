package com.iotSmartTrash.service;

import com.iotSmartTrash.dto.BinRealtimeStatusResponseDTO;
import com.iotSmartTrash.exception.ServiceException;
import com.iotSmartTrash.model.BinMetadata;
import com.iotSmartTrash.model.BinRawSensorLog;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * Service trạng thái thùng rác theo mô hình non-realtime:
 * trạng thái hiện tại được suy ra từ raw sensor logs mới nhất.
 */
@Service
@RequiredArgsConstructor
public class BinRealtimeService {

    private static final long OFFLINE_THRESHOLD_MS = 2 * 60 * 1000L;

    private final BinRawSensorLogService rawSensorLogService;
    private final BinMetadataService binMetadataService;

    /**
     * Lấy trạng thái hiện tại của 1 thùng rác từ raw log mới nhất.
     */
    public BinRealtimeStatusResponseDTO getStatus(String binId) {
        binMetadataService.getBinById(binId);

        List<BinRawSensorLog> logs = rawSensorLogService.getRecentLogsForBin(binId, 1);
        BinRawSensorLog latest = logs.isEmpty() ? null : logs.get(0);
        return BinRealtimeStatusResponseDTO.fromRawLog(binId, latest, OFFLINE_THRESHOLD_MS);
    }

    /**
     * Lấy trạng thái hiện tại của tất cả thùng rác theo metadata.
     */
    public List<BinRealtimeStatusResponseDTO> getAllStatuses() {
        try {
            List<BinRealtimeStatusResponseDTO> statuses = new ArrayList<>();
            List<BinMetadata> bins = binMetadataService.getAllBins();
            for (BinMetadata bin : bins) {
                String binId = bin.getId();
                List<BinRawSensorLog> logs = rawSensorLogService.getRecentLogsForBin(binId, 1);
                BinRawSensorLog latest = logs.isEmpty() ? null : logs.get(0);
                statuses.add(BinRealtimeStatusResponseDTO.fromRawLog(binId, latest, OFFLINE_THRESHOLD_MS));
            }
            return statuses;
        } catch (Exception e) {
            throw new ServiceException("Cannot get all bin statuses from raw logs", e);
        }
    }
}

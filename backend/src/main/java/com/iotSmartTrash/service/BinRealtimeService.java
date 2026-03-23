package com.iotSmartTrash.service;

import com.iotSmartTrash.exception.ServiceException;
import com.iotSmartTrash.model.BinMetadata;
import com.iotSmartTrash.model.BinRawSensorLog;
import com.iotSmartTrash.model.BinRealtimeStatus;
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
    public BinRealtimeStatus getStatus(String binId) {
        // Ensure bin exists in metadata; otherwise return 404 as before.
        binMetadataService.getBinById(binId);

        List<BinRawSensorLog> logs = rawSensorLogService.getRecentLogsForBin(binId, 1);
        if (logs.isEmpty()) {
            return BinRealtimeStatus.builder()
                    .id(binId)
                    .status("UNKNOWN")
                    .temperature(0.0)
                    .batteryLevel(0)
                    .fillOrganic(0)
                    .fillRecycle(0)
                    .fillNonRecycle(0)
                    .fillHazardous(0)
                    .lastUpdated(0L)
                    .build();
        }
        return toStatus(binId, logs.get(0));
    }

    /**
     * Lấy trạng thái hiện tại của tất cả thùng rác theo metadata.
     */
    public List<BinRealtimeStatus> getAllStatuses() {
        try {
            List<BinRealtimeStatus> statuses = new ArrayList<>();
            List<BinMetadata> bins = binMetadataService.getAllBins();
            for (BinMetadata bin : bins) {
                String binId = bin.getId();
                List<BinRawSensorLog> logs = rawSensorLogService.getRecentLogsForBin(binId, 1);
                if (logs.isEmpty()) {
                    statuses.add(BinRealtimeStatus.builder()
                            .id(binId)
                            .status("UNKNOWN")
                            .temperature(0.0)
                            .batteryLevel(0)
                            .fillOrganic(0)
                            .fillRecycle(0)
                            .fillNonRecycle(0)
                            .fillHazardous(0)
                            .lastUpdated(0L)
                            .build());
                    continue;
                }
                statuses.add(toStatus(binId, logs.get(0)));
            }
            return statuses;
        } catch (Exception e) {
            throw new ServiceException("Cannot get all bin statuses from raw logs", e);
        }
    }

    private BinRealtimeStatus toStatus(String binId, BinRawSensorLog log) {
        long lastUpdated = log.getRecordedAt() != null ? log.getRecordedAt() : 0L;
        long ageMs = System.currentTimeMillis() - lastUpdated;
        String status = (lastUpdated > 0 && ageMs <= OFFLINE_THRESHOLD_MS) ? "ONLINE" : "OFFLINE";

        return BinRealtimeStatus.builder()
                .id(binId)
                .status(status)
                .temperature(log.getTemperature())
                .batteryLevel(log.getBatteryLevel())
                .fillOrganic(log.getFillOrganic())
                .fillRecycle(log.getFillRecycle())
                .fillNonRecycle(log.getFillNonRecycle())
                .fillHazardous(log.getFillHazardous())
                .lastUpdated(lastUpdated)
                .build();
    }
}

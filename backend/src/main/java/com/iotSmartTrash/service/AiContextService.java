package com.iotSmartTrash.service;

import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.*;

@Service
@RequiredArgsConstructor
public class AiContextService {

    private final BinMetadataService binMetadataService;
    private final BinRawSensorLogService binRawSensorLogService;
    private final BinScheduleService binScheduleService;

    public Map<String, Object> getAiContext() {
        List<Map<String, Object>> bins = new ArrayList<>();

        binMetadataService.getAllBins().forEach(bin -> {
            Map<String, Object> item = new LinkedHashMap<>();

            item.put("binId", bin.getId());
            item.put("name", bin.getName());
            item.put("locationDescription", bin.getLocationDescription());
            item.put("latitude", bin.getLatitude());
            item.put("longitude", bin.getLongitude());
            item.put("installedAt", bin.getInstalledAt());

            try {
                var latestLog = binRawSensorLogService.getLatestRawLogByBinId(bin.getId());

                int fillOrganic = Optional.ofNullable(latestLog.getFillOrganic()).orElse(0);
                int fillRecycle = Optional.ofNullable(latestLog.getFillRecycle()).orElse(0);
                int fillNonRecycle = Optional.ofNullable(latestLog.getFillNonRecycle()).orElse(0);
                int fillHazardous = Optional.ofNullable(latestLog.getFillHazardous()).orElse(0);
                int batteryLevel = Optional.ofNullable(latestLog.getBatteryLevel()).orElse(0);

                int maxFill = Collections.max(List.of(
                        fillOrganic,
                        fillRecycle,
                        fillNonRecycle,
                        fillHazardous
                ));

                item.put("batteryLevel", batteryLevel);
                item.put("fillOrganic", fillOrganic);
                item.put("fillRecycle", fillRecycle);
                item.put("fillNonRecycle", fillNonRecycle);
                item.put("fillHazardous", fillHazardous);
                item.put("recordedAt", latestLog.getRecordedAt());
                item.put("maxFill", maxFill);
                item.put("nearlyFull", maxFill >= 80);

            } catch (Exception e) {
                item.put("sensorData", "No latest sensor log found");
                item.put("nearlyFull", false);
                item.put("maxFill", 0);
            }

            bins.add(item);
        });

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("bins", bins);

        try {
            result.put("pickupSchedule", binScheduleService.getPickupSchedule(40));
        } catch (Exception e) {
            result.put("pickupSchedule", Collections.emptyList());
        }

        return result;
    }
}
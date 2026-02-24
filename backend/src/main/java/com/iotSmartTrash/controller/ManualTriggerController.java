package com.iotSmartTrash.controller;

import com.iotSmartTrash.scheduler.SensorLogScheduler;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@RestController
@RequestMapping("/api/v1/trigger")
@RequiredArgsConstructor
public class ManualTriggerController {

    private final SensorLogScheduler scheduler;

    /**
     * Triggers sensor log aggregation in background.
     * HTTP returns immediately (202 Accepted) without waiting for completion.
     */
    @PostMapping("/aggregate-sensor-logs")
    public ResponseEntity<String> triggerAggregateLogs() {
        Thread.ofVirtual().name("scheduler-trigger").start(() -> {
            try {
                scheduler.aggregateSensorLogs();
            } catch (Exception e) {
                // Virtual thread exceptions are silent by default — log explicitly
                log.error("[ManualTrigger] aggregateSensorLogs failed: {}", e.getMessage(), e);
            }
        });
        return ResponseEntity.accepted()
                .body("Triggered aggregate sensor logs in background. Check console for progress.");
    }
}

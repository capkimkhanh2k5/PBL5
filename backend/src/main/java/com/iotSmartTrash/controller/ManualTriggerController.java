package com.iotSmartTrash.controller;

import com.iotSmartTrash.scheduler.SensorLogScheduler;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/trigger")
@RequiredArgsConstructor
public class ManualTriggerController {

    private final SensorLogScheduler scheduler;

    /**
     * Chạy async trong background thread, HTTP trả về ngay lập tức.
     */
    @PostMapping("/aggregate-sensor-logs")
    public ResponseEntity<String> triggerAggregateLogs() {
        // Virtual Thread
        Thread.ofVirtual().name("scheduler-trigger").start(scheduler::aggregateSensorLogs);
        return ResponseEntity.accepted()
                .body("Triggered aggregate sensor logs in background. Read console to see progress.");
    }
}

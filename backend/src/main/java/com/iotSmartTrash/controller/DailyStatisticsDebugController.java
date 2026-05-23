package com.iotSmartTrash.controller;

import com.iotSmartTrash.scheduler.DailyStatisticsScheduler;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.time.LocalDate;
import java.time.ZoneId;

@RestController
@RequiredArgsConstructor
public class DailyStatisticsDebugController {

    private static final ZoneId ZONE = ZoneId.of("Asia/Ho_Chi_Minh");

    private final DailyStatisticsScheduler dailyStatisticsScheduler;

    @GetMapping("/api/debug/daily-statistics/today")
    public String aggregateToday() {
        LocalDate today = LocalDate.now(ZONE);
        dailyStatisticsScheduler.aggregateDailyStatisticsForDate(today);
        return "Triggered daily statistics for today: " + today;
    }

    @GetMapping("/api/debug/daily-statistics/yesterday")
    public String aggregateYesterday() {
        LocalDate yesterday = LocalDate.now(ZONE).minusDays(1);
        dailyStatisticsScheduler.aggregateDailyStatisticsForDate(yesterday);
        return "Triggered daily statistics for yesterday: " + yesterday;
    }

    @GetMapping("/api/debug/daily-statistics/range")
    public String aggregateRange(
            @RequestParam String startDate,
            @RequestParam String endDate
    ) {
        LocalDate start = LocalDate.parse(startDate);
        LocalDate end = LocalDate.parse(endDate);

        int count = 0;

        LocalDate current = start;
        while (!current.isAfter(end)) {
            dailyStatisticsScheduler.aggregateDailyStatisticsForDate(current);
            count++;
            current = current.plusDays(1);
        }

        return "Triggered daily statistics from "
                + start
                + " to "
                + end
                + ". Total days: "
                + count;
    }
}
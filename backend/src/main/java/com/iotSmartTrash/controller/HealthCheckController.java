package com.iotSmartTrash.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/health")
public class HealthCheckController {

    @GetMapping
    public String checkHealth() {
        return "Smart Trash Backend is running! (Spring Boot + Firebase Admin SDK)";
    }
}

package com.iotSmartTrash.controller;

import com.iotSmartTrash.service.AiContextService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/v1/ai")
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
public class AiContextController {

    private final AiContextService aiContextService;

    @GetMapping("/context")
    public Map<String, Object> getAiContext() {
        return aiContextService.getAiContext();
    }
}
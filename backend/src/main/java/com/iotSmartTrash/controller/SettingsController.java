package com.iotSmartTrash.controller;

import com.iotSmartTrash.dto.UpdateUsernameRequest;
import com.iotSmartTrash.model.User;
import com.iotSmartTrash.service.SettingsService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/settings")
@RequiredArgsConstructor
public class SettingsController {

    private final SettingsService settingsService;

    @GetMapping("/me")
    public User getProfile() {
        return settingsService.getCurrentUser();
    }

    @PutMapping("/username")
    public void updateUsername(@RequestBody UpdateUsernameRequest request) {
        settingsService.updateUsername(request.getUsername());
    }
}
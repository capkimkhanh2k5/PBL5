package com.iotSmartTrash.controller;

import com.iotSmartTrash.dto.UserResponseDTO;
import com.iotSmartTrash.dto.UserRoleUpdateDTO;
import com.iotSmartTrash.model.User;
import com.iotSmartTrash.service.UserService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/v1/users")
@RequiredArgsConstructor
@Validated
public class UserController {

    private final UserService userService;

    /**
     * Lấy thông tin user hiện tại dựa trên Firebase UID từ token.
     */
    @GetMapping("/me")
    public ResponseEntity<UserResponseDTO> getCurrentUser(Authentication authentication) {
        String uid = (String) authentication.getPrincipal();
        User user = userService.createUserIfNotExists(uid);
        return ResponseEntity.ok(UserResponseDTO.fromModel(user));
    }

    /**
     * Client gọi sau khi đăng ký/đăng nhập Firebase Auth thành công
     * để đồng bộ thông tin user vào Firestore.
     */
    @PostMapping("/sync")
    public ResponseEntity<UserResponseDTO> syncUser(Authentication authentication) {
        String uid = (String) authentication.getPrincipal();
        User user = userService.createUserIfNotExists(uid);
        return ResponseEntity.ok(UserResponseDTO.fromModel(user));
    }

    @GetMapping
    public ResponseEntity<List<UserResponseDTO>> getAllUsers() {
        List<User> users = userService.getAllUsers();
        List<UserResponseDTO> dtos = users.stream()
                .map(UserResponseDTO::fromModel)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }

    @GetMapping("/{uid}")
    public ResponseEntity<UserResponseDTO> getUserById(@PathVariable String uid) {
        User user = userService.getUserById(uid);
        return ResponseEntity.ok(UserResponseDTO.fromModel(user));
    }

    @PatchMapping("/{uid}/role")
    public ResponseEntity<String> updateUserRole(
            @PathVariable String uid,
            @Valid @RequestBody UserRoleUpdateDTO body) {
        String updateTime = userService.updateUserRole(uid, body.getRole().name());
        return ResponseEntity.ok("Successfully updated user role at " + updateTime);
    }
}

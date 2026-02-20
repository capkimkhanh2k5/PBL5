package com.iotSmartTrash.controller;

import com.iotSmartTrash.model.User;
import com.iotSmartTrash.service.UserService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

@RestController
@RequestMapping("/api/v1/users")
@RequiredArgsConstructor
public class UserController {

    private final UserService userService;

    @GetMapping
    public ResponseEntity<List<User>> getAllUsers() throws ExecutionException, InterruptedException {
        return ResponseEntity.ok(userService.getAllUsers());
    }

    @GetMapping("/{uid}")
    public ResponseEntity<User> getUserById(@PathVariable String uid) throws ExecutionException, InterruptedException {
        User user = userService.getUserById(uid);
        if (user != null) {
            return ResponseEntity.ok(user);
        }
        return ResponseEntity.notFound().build();
    }

    @PatchMapping("/{uid}/role")
    public ResponseEntity<String> updateUserRole(@PathVariable String uid, @RequestBody Map<String, String> body)
            throws ExecutionException, InterruptedException {
        String role = body.get("role");
        String updateTime = userService.updateUserRole(uid, role);
        return ResponseEntity.ok("Successfully updated user role at " + updateTime);
    }
}

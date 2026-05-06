package com.iotSmartTrash.service;

import com.google.api.core.ApiFuture;
import com.google.cloud.firestore.DocumentSnapshot;
import com.google.cloud.firestore.Firestore;
import com.iotSmartTrash.model.User;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class SettingsService {

    private final Firestore firestore;

    public User getCurrentUser() {
        String uid = getCurrentUid();

        try {
            ApiFuture<DocumentSnapshot> future = firestore.collection("users")
                    .document(uid)
                    .get();

            DocumentSnapshot document = future.get();

            if (!document.exists()) {
                throw new RuntimeException("User not found");
            }

            User user = document.toObject(User.class);
            if (user == null) {
                throw new RuntimeException("Cannot parse user data");
            }

            user.setUid(uid);
            return user;

        } catch (Exception e) {
            throw new RuntimeException("Cannot get current user", e);
        }
    }

    public void updateUsername(String username) {
        String uid = getCurrentUid();

        if (username == null || username.trim().isEmpty()) {
            throw new RuntimeException("Username cannot be empty");
        }

        try {
            firestore.collection("users")
                    .document(uid)
                    .update("username", username.trim())
                    .get();
        } catch (Exception e) {
            throw new RuntimeException("Cannot update username", e);
        }
    }

    private String getCurrentUid() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();

        if (authentication == null || !authentication.isAuthenticated()) {
            throw new RuntimeException("User not authenticated");
        }

        return authentication.getName();
    }
}
package com.iotSmartTrash.service;

import com.google.cloud.Timestamp;
import com.google.cloud.firestore.DocumentReference;
import com.google.cloud.firestore.DocumentSnapshot;
import com.google.cloud.firestore.Firestore;
import com.google.cloud.firestore.QueryDocumentSnapshot;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseAuthException;
import com.google.firebase.auth.UserRecord;
import com.iotSmartTrash.exception.ResourceNotFoundException;
import com.iotSmartTrash.exception.ServiceException;
import com.iotSmartTrash.model.User;
import com.iotSmartTrash.model.enums.UserRole;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

@Slf4j
@Service
@RequiredArgsConstructor
public class UserService {

    private static final String COLLECTION_NAME = "users";

    private final Firestore firestore;

    /**
     * Tự động tạo User trong Firestore nếu uid chưa tồn tại (đăng nhập lần đầu).
     * Lấy thông tin email, displayName từ Firebase Auth để lưu vào Firestore.
     * Trả về User đã tồn tại hoặc User vừa được tạo mới.
     */
    public User createUserIfNotExists(String uid) {
        try {
            DocumentReference docRef = firestore.collection(COLLECTION_NAME).document(uid);
            DocumentSnapshot doc = docRef.get().get();

            if (doc.exists()) {
                User user = doc.toObject(User.class);
                if (user != null) {
                    user.setUid(doc.getId());
                    return user;
                }
            }

            // Lấy thông tin user từ Firebase Auth
            String email = "";
            String displayName = "";
            String photoUrl = "";
            try {
                UserRecord userRecord = FirebaseAuth.getInstance().getUser(uid);
                email = userRecord.getEmail() != null ? userRecord.getEmail() : "";
                displayName = userRecord.getDisplayName() != null ? userRecord.getDisplayName() : "";
                photoUrl = userRecord.getPhotoUrl() != null ? userRecord.getPhotoUrl() : "";
            } catch (FirebaseAuthException e) {
                log.warn("Cannot fetch Firebase Auth info for uid={}: {}", uid, e.getMessage());
            }

            // Tạo user mới với role mặc định là USER
            Map<String, Object> userData = new HashMap<>();
            userData.put("username", displayName);
            userData.put("email", email);
            userData.put("role", UserRole.USER.name());
            userData.put("avatar_url", photoUrl);
            userData.put("created_at", Timestamp.now());

            docRef.set(userData).get();
            log.info("Created new user in Firestore: uid={}, email={}", uid, email);

            User newUser = User.builder()
                    .uid(uid)
                    .username(displayName)
                    .email(email)
                    .role(UserRole.USER)
                    .avatarUrl(photoUrl)
                    .createdAt(Timestamp.now())
                    .build();
            return newUser;

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot create user: operation interrupted", e);
        } catch (ExecutionException e) {
            throw new ServiceException("Cannot create user: " + uid, e.getCause());
        }
    }

    /**
     * Lấy role của user theo uid, trả về UserRole.USER nếu user chưa tồn tại.
     * Được dùng bởi FirebaseTokenFilter để gán role vào SecurityContext.
     */
    public UserRole getUserRoleByUid(String uid) {
        try {
            DocumentReference docRef = firestore.collection(COLLECTION_NAME).document(uid);
            DocumentSnapshot doc = docRef.get().get();
            if (!doc.exists()) {
                return UserRole.USER;
            }
            Object roleObj = doc.get("role");
            if (roleObj == null) {
                return UserRole.USER;
            }
            return UserRole.valueOf(roleObj.toString());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            log.error("Cannot get user role: operation interrupted", e);
            return UserRole.USER;
        } catch (ExecutionException e) {
            log.error("Cannot get user role for uid={}", uid, e);
            return UserRole.USER;
        } catch (IllegalArgumentException e) {
            log.warn("Invalid role value in Firestore for uid={}", uid);
            return UserRole.USER;
        }
    }

    public List<User> getAllUsers() {
        try {
            List<User> users = new ArrayList<>();
            for (QueryDocumentSnapshot doc : firestore.collection(COLLECTION_NAME).get().get().getDocuments()) {
                User user = doc.toObject(User.class);
                user.setUid(doc.getId());
                users.add(user);
            }
            return users;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot get list of users: operation interrupted", e);
        } catch (ExecutionException e) {
            throw new ServiceException("Cannot get list of users", e.getCause());
        }
    }

    public User getUserById(String uid) {
        try {
            DocumentReference docRef = firestore.collection(COLLECTION_NAME).document(uid);
            DocumentSnapshot doc = docRef.get().get();
            if (!doc.exists()) {
                throw new ResourceNotFoundException("User", uid);
            }
            User user = doc.toObject(User.class);
            user.setUid(doc.getId());
            return user;
        } catch (ResourceNotFoundException e) {
            throw e;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot get user information: operation interrupted", e);
        } catch (ExecutionException e) {
            throw new ServiceException("Cannot get user information: " + uid, e.getCause());
        }
    }

    public String updateUserRole(String uid, String role) {
        try {
            return firestore.collection(COLLECTION_NAME).document(uid)
                    .update("role", role)
                    .get()
                    .getUpdateTime()
                    .toString();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot update role for user: operation interrupted", e);
        } catch (ExecutionException e) {
            throw new ServiceException("Cannot update role for user: " + uid, e.getCause());
        }
    }
}

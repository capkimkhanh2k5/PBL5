package com.iotSmartTrash.service;

import com.google.cloud.firestore.DocumentReference;
import com.google.cloud.firestore.Firestore;
import com.google.cloud.firestore.QueryDocumentSnapshot;
import com.iotSmartTrash.exception.ResourceNotFoundException;
import com.iotSmartTrash.exception.ServiceException;
import com.iotSmartTrash.model.User;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

@Service
@RequiredArgsConstructor
public class UserService {

    private static final String COLLECTION_NAME = "users";

    private final Firestore firestore;

    public List<User> getAllUsers() {
        try {
            List<User> users = new ArrayList<>();
            for (QueryDocumentSnapshot doc : firestore.collection(COLLECTION_NAME).get().get().getDocuments()) {
                User user = doc.toObject(User.class);
                user.setUid(doc.getId());
                users.add(user);
            }
            return users;
        } catch (Exception e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot get list of users", e);
        }
    }

    public User getUserById(String uid) {
        try {
            DocumentReference docRef = firestore.collection(COLLECTION_NAME).document(uid);
            com.google.cloud.firestore.DocumentSnapshot doc = docRef.get().get();
            if (!doc.exists()) {
                throw new ResourceNotFoundException("User", uid);
            }
            User user = doc.toObject(User.class);
            user.setUid(doc.getId());
            return user;
        } catch (ResourceNotFoundException e) {
            throw e;
        } catch (Exception e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot get user information: " + uid, e);
        }
    }

    public String updateUserRole(String uid, String role) {
        try {
            return firestore.collection(COLLECTION_NAME).document(uid)
                    .update("role", role)
                    .get()
                    .getUpdateTime()
                    .toString();
        } catch (Exception e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot update role for user: " + uid, e);
        }
    }
}

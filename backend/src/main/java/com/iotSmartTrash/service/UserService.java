package com.iotSmartTrash.service;

import com.google.api.core.ApiFuture;
import com.google.cloud.firestore.DocumentReference;
import com.google.cloud.firestore.Firestore;
import com.google.cloud.firestore.QueryDocumentSnapshot;
import com.google.cloud.firestore.WriteResult;
import com.google.firebase.cloud.FirestoreClient;
import com.iotSmartTrash.model.User;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

@Service
public class UserService {

    private static final String COLLECTION_NAME = "users";

    public List<User> getAllUsers() throws ExecutionException, InterruptedException {
        Firestore dbFirestore = FirestoreClient.getFirestore();
        List<User> users = new ArrayList<>();

        for (QueryDocumentSnapshot document : dbFirestore.collection(COLLECTION_NAME).get().get().getDocuments()) {
            User user = document.toObject(User.class);
            user.setUid(document.getId());
            users.add(user);
        }
        return users;
    }

    public User getUserById(String uid) throws ExecutionException, InterruptedException {
        Firestore dbFirestore = FirestoreClient.getFirestore();
        DocumentReference documentReference = dbFirestore.collection(COLLECTION_NAME).document(uid);
        com.google.cloud.firestore.DocumentSnapshot document = documentReference.get().get();

        if (document.exists()) {
            User user = document.toObject(User.class);
            user.setUid(document.getId());
            return user;
        }
        return null;
    }

    public String updateUserRole(String uid, String role) throws ExecutionException, InterruptedException {
        Firestore dbFirestore = FirestoreClient.getFirestore();
        ApiFuture<WriteResult> collectionsApiFuture = dbFirestore.collection(COLLECTION_NAME).document(uid)
                .update("role", role);
        return collectionsApiFuture.get().getUpdateTime().toString();
    }
}

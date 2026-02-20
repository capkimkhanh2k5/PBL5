package com.iotSmartTrash.service;

import com.google.api.core.ApiFuture;
import com.google.cloud.firestore.DocumentReference;
import com.google.cloud.firestore.Firestore;
import com.google.cloud.firestore.WriteResult;
import com.google.firebase.cloud.FirestoreClient;
import com.iotSmartTrash.model.ClassificationLog;
import org.springframework.stereotype.Service;
import java.util.concurrent.ExecutionException;

@Service
public class ClassificationLogService {

    private static final String COLLECTION_NAME = "classification_logs";

    public String saveLog(ClassificationLog log) throws ExecutionException, InterruptedException {
        Firestore dbFirestore = FirestoreClient.getFirestore();
        DocumentReference docRef = dbFirestore.collection(COLLECTION_NAME).document();
        log.setLog_id(docRef.getId()); // Tự sinh ID
        log.setClassified_at(com.google.cloud.Timestamp.now());

        ApiFuture<WriteResult> collectionsApiFuture = docRef.set(log);
        return collectionsApiFuture.get().getUpdateTime().toString();
    }
}

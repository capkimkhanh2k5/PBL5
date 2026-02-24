package com.iotSmartTrash.service;

import com.google.cloud.Timestamp;
import com.google.cloud.firestore.DocumentReference;
import com.google.cloud.firestore.Firestore;
import com.iotSmartTrash.exception.ServiceException;
import com.iotSmartTrash.model.ClassificationLog;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.concurrent.ExecutionException;

@Service
@RequiredArgsConstructor
public class ClassificationLogService {

    private static final String COLLECTION_NAME = "classification_logs";

    private final Firestore firestore;

    public String saveLog(ClassificationLog log) {
        try {
            DocumentReference docRef = firestore.collection(COLLECTION_NAME).document();
            log.setLogId(docRef.getId());
            log.setClassifiedAt(Timestamp.now());
            return docRef.set(log).get().getUpdateTime().toString();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot save classification log: operation interrupted", e);
        } catch (ExecutionException e) {
            throw new ServiceException("Cannot save classification log", e.getCause());
        }
    }
}

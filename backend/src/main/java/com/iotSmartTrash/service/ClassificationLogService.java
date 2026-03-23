package com.iotSmartTrash.service;

import com.google.cloud.Timestamp;
import com.google.cloud.firestore.DocumentReference;
import com.google.cloud.firestore.DocumentSnapshot;
import com.google.cloud.firestore.Query;
import com.google.cloud.firestore.QueryDocumentSnapshot;
import com.google.cloud.firestore.Firestore;
import com.iotSmartTrash.exception.ServiceException;
import com.iotSmartTrash.model.ClassificationLog;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
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

            // Persist canonical snake_case keys for stable reads/queries.
            Map<String, Object> payload = new HashMap<>();
            payload.put("log_id", log.getLogId());
            payload.put("bin_id", log.getBinId());
            payload.put("image_url", log.getImageUrl());
            payload.put("classification_result", log.getClassificationResult());
            payload.put("confidence_score", log.getConfidenceScore());
            payload.put("classified_at", log.getClassifiedAt());

            return docRef.set(payload).get().getUpdateTime().toString();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot save classification log: operation interrupted", e);
        } catch (ExecutionException e) {
            throw new ServiceException("Cannot save classification log", e.getCause());
        }
    }

    public List<ClassificationLog> getLatestLogs(String binId, int limit) {
        try {
            List<ClassificationLog> logs = new ArrayList<>();
            Query query;

            if (binId != null && !binId.isBlank()) {
                String normalizedBinId = _normalizeBinId(binId);

                // Avoid composite-index requirement by filtering first, then sorting in memory.
                int fetchLimit = Math.max(limit * 5, 100);
                Query snakeCaseQuery = firestore.collection(COLLECTION_NAME)
                    .whereEqualTo("bin_id", binId)
                        .limit(fetchLimit);

                Query camelCaseQuery = firestore.collection(COLLECTION_NAME)
                    .whereEqualTo("binId", binId)
                        .limit(fetchLimit);

                Set<String> seenDocIds = new java.util.HashSet<>();

                for (QueryDocumentSnapshot doc : snakeCaseQuery.get().get().getDocuments()) {
                    ClassificationLog log = mapClassificationLog(doc);
                    if (seenDocIds.add(doc.getId())) {
                        logs.add(log);
                    }
                }

                for (QueryDocumentSnapshot doc : camelCaseQuery.get().get().getDocuments()) {
                    ClassificationLog log = mapClassificationLog(doc);
                    if (seenDocIds.add(doc.getId())) {
                        logs.add(log);
                    }
                }

                if (logs.isEmpty()) {
                    // Fallback: fetch latest docs and match bin id after normalization.
                    Query fallbackQuery = firestore.collection(COLLECTION_NAME)
                            .orderBy("classified_at", Query.Direction.DESCENDING)
                            .limit(fetchLimit);

                    for (QueryDocumentSnapshot doc : fallbackQuery.get().get().getDocuments()) {
                        ClassificationLog log = mapClassificationLog(doc);
                        if (_normalizeBinId(log.getBinId()).equals(normalizedBinId)) {
                            logs.add(log);
                        }
                    }
                }
            } else {
                query = firestore.collection(COLLECTION_NAME)
                        .orderBy("classified_at", Query.Direction.DESCENDING)
                        .limit(limit);

                for (QueryDocumentSnapshot doc : query.get().get().getDocuments()) {
                    logs.add(mapClassificationLog(doc));
                }
            }

            if (binId != null && !binId.isBlank()) {
                logs.sort((a, b) -> Long.compare(_classifiedAtMillis(b), _classifiedAtMillis(a)));
                if (logs.size() > limit) {
                    return new ArrayList<>(logs.subList(0, limit));
                }
            }

            return logs;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot get classification logs: operation interrupted", e);
        } catch (ExecutionException e) {
            throw new ServiceException("Cannot get classification logs", e.getCause());
        }
    }

    private ClassificationLog mapClassificationLog(DocumentSnapshot doc) {
        String logId = _readString(doc, "log_id", "logId");
        if (logId == null || logId.isBlank()) {
            logId = doc.getId();
        }

        return ClassificationLog.builder()
                .logId(logId)
                .binId(_readString(doc, "bin_id", "binId"))
                .imageUrl(_readString(doc, "image_url", "imageUrl"))
                .classificationResult(_readString(doc, "classification_result", "classificationResult"))
                .confidenceScore(_readDouble(doc, "confidence_score", "confidenceScore"))
                .classifiedAt(_readTimestamp(doc, "classified_at", "classifiedAt"))
                .build();
    }

    private long _classifiedAtMillis(ClassificationLog log) {
        if (log == null || log.getClassifiedAt() == null) {
            return Long.MIN_VALUE;
        }
        return log.getClassifiedAt().toDate().getTime();
    }

    private String _readString(DocumentSnapshot doc, String primary, String fallback) {
        Object value = doc.get(primary);
        if (value == null) {
            value = doc.get(fallback);
        }
        return value != null ? value.toString() : null;
    }

    private Double _readDouble(DocumentSnapshot doc, String primary, String fallback) {
        Object value = doc.get(primary);
        if (value == null) {
            value = doc.get(fallback);
        }
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        return null;
    }

    private Timestamp _readTimestamp(DocumentSnapshot doc, String primary, String fallback) {
        Object value = doc.get(primary);
        if (value == null) {
            value = doc.get(fallback);
        }
        if (value instanceof Timestamp ts) {
            return ts;
        }
        return null;
    }

    private String _normalizeBinId(String value) {
        if (value == null) {
            return "";
        }
        return value.trim().toLowerCase();
    }
}

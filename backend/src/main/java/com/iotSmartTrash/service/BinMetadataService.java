package com.iotSmartTrash.service;

import com.google.api.core.ApiFuture;
import com.google.cloud.firestore.DocumentReference;
import com.google.cloud.firestore.Firestore;
import com.google.cloud.firestore.QueryDocumentSnapshot;
import com.google.cloud.firestore.WriteResult;
import com.google.firebase.cloud.FirestoreClient;
import com.iotSmartTrash.model.BinMetadata;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

@Service
public class BinMetadataService {

    private static final String COLLECTION_NAME = "bins_metadata";

    public List<BinMetadata> getAllBins() throws ExecutionException, InterruptedException {
        Firestore dbFirestore = FirestoreClient.getFirestore();
        List<BinMetadata> bins = new ArrayList<>();

        for (QueryDocumentSnapshot document : dbFirestore.collection(COLLECTION_NAME).get().get().getDocuments()) {
            BinMetadata bin = document.toObject(BinMetadata.class);
            bin.setId(document.getId());
            bins.add(bin);
        }
        return bins;
    }

    public BinMetadata getBinById(String documentId) throws ExecutionException, InterruptedException {
        Firestore dbFirestore = FirestoreClient.getFirestore();
        DocumentReference documentReference = dbFirestore.collection(COLLECTION_NAME).document(documentId);
        com.google.cloud.firestore.DocumentSnapshot document = documentReference.get().get();

        if (document.exists()) {
            BinMetadata bin = document.toObject(BinMetadata.class);
            bin.setId(document.getId());
            return bin;
        }
        return null; // Không tìm thấy
    }

    public String createBin(BinMetadata bin) throws ExecutionException, InterruptedException {
        Firestore dbFirestore = FirestoreClient.getFirestore();
        DocumentReference docRef;
        if (bin.getId() != null && !bin.getId().isEmpty()) {
            docRef = dbFirestore.collection(COLLECTION_NAME).document(bin.getId());
        } else {
            docRef = dbFirestore.collection(COLLECTION_NAME).document();
        }

        ApiFuture<WriteResult> collectionsApiFuture = docRef.set(bin);
        return collectionsApiFuture.get().getUpdateTime().toString();
    }

    public String updateBin(String binId, BinMetadata bin) throws ExecutionException, InterruptedException {
        Firestore dbFirestore = FirestoreClient.getFirestore();
        ApiFuture<WriteResult> collectionsApiFuture = dbFirestore.collection(COLLECTION_NAME).document(binId).set(bin);
        return collectionsApiFuture.get().getUpdateTime().toString();
    }

    public String deleteBin(String binId) throws ExecutionException, InterruptedException {
        Firestore dbFirestore = FirestoreClient.getFirestore();
        ApiFuture<WriteResult> writeResult = dbFirestore.collection(COLLECTION_NAME).document(binId).delete();
        return writeResult.get().getUpdateTime().toString();
    }
}

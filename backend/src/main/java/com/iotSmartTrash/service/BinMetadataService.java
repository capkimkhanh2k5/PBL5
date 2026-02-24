package com.iotSmartTrash.service;

import com.google.cloud.firestore.DocumentReference;
import com.google.cloud.firestore.Firestore;
import com.google.cloud.firestore.QueryDocumentSnapshot;
import com.iotSmartTrash.exception.ResourceNotFoundException;
import com.iotSmartTrash.exception.ServiceException;
import com.iotSmartTrash.model.BinMetadata;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

@Service
@RequiredArgsConstructor
public class BinMetadataService {

    private static final String COLLECTION_NAME = "bins_metadata";

    private final Firestore firestore;

    public List<BinMetadata> getAllBins() {
        try {
            List<BinMetadata> bins = new ArrayList<>();
            for (QueryDocumentSnapshot doc : firestore.collection(COLLECTION_NAME).get().get().getDocuments()) {
                BinMetadata bin = doc.toObject(BinMetadata.class);
                bin.setId(doc.getId());
                bins.add(bin);
            }
            return bins;
        } catch (Exception e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot get list of bins", e);
        }
    }

    public BinMetadata getBinById(String binId) {
        try {
            com.google.cloud.firestore.DocumentSnapshot doc = firestore.collection(COLLECTION_NAME).document(binId)
                    .get().get();
            if (!doc.exists()) {
                throw new ResourceNotFoundException("Bin", binId);
            }
            BinMetadata bin = doc.toObject(BinMetadata.class);
            bin.setId(doc.getId());
            return bin;
        } catch (ResourceNotFoundException e) {
            throw e;
        } catch (Exception e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot get bin information: " + binId, e);
        }
    }

    /** Tạo mới bin — ID luôn do Firestore auto-generate, không cho client set ID */
    public String createBin(BinMetadata bin) {
        try {
            DocumentReference docRef = firestore.collection(COLLECTION_NAME).document();
            bin.setId(docRef.getId());
            bin.setInstalledAt(com.google.cloud.Timestamp.now());
            return docRef.set(bin).get().getUpdateTime().toString();
        } catch (Exception e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot create bin", e);
        }
    }

    public String updateBin(String binId, BinMetadata bin) {
        try {
            return firestore.collection(COLLECTION_NAME).document(binId)
                    .set(bin).get().getUpdateTime().toString();
        } catch (Exception e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot update bin: " + binId, e);
        }
    }

    public String deleteBin(String binId) {
        try {
            return firestore.collection(COLLECTION_NAME).document(binId)
                    .delete().get().getUpdateTime().toString();
        } catch (Exception e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot delete bin: " + binId, e);
        }
    }
}

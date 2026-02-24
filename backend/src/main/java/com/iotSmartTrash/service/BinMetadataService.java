package com.iotSmartTrash.service;

import com.google.cloud.Timestamp;
import com.google.cloud.firestore.DocumentSnapshot;
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
import java.util.concurrent.ExecutionException;

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
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot get list of bins: operation interrupted", e);
        } catch (ExecutionException e) {
            throw new ServiceException("Cannot get list of bins", e.getCause());
        }
    }

    public BinMetadata getBinById(String binId) {
        try {
            DocumentSnapshot doc = firestore.collection(COLLECTION_NAME).document(binId)
                    .get().get();
            if (!doc.exists()) {
                throw new ResourceNotFoundException("Bin", binId);
            }
            BinMetadata bin = doc.toObject(BinMetadata.class);
            bin.setId(doc.getId());
            return bin;
        } catch (ResourceNotFoundException e) {
            throw e;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot get bin information: operation interrupted", e);
        } catch (ExecutionException e) {
            throw new ServiceException("Cannot get bin information: " + binId, e.getCause());
        }
    }

    /**
     * Create a new bin — ID is always Firestore auto-generated, clients cannot set
     * it
     */
    public String createBin(BinMetadata bin) {
        try {
            DocumentReference docRef = firestore.collection(COLLECTION_NAME).document();
            bin.setId(docRef.getId());
            bin.setInstalledAt(Timestamp.now());
            return docRef.set(bin).get().getUpdateTime().toString();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot create bin: operation interrupted", e);
        } catch (ExecutionException e) {
            throw new ServiceException("Cannot create bin", e.getCause());
        }
    }

    public String updateBin(String binId, BinMetadata bin) {
        try {
            return firestore.collection(COLLECTION_NAME).document(binId)
                    .set(bin).get().getUpdateTime().toString();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot update bin: operation interrupted", e);
        } catch (ExecutionException e) {
            throw new ServiceException("Cannot update bin: " + binId, e.getCause());
        }
    }

    public String deleteBin(String binId) {
        try {
            return firestore.collection(COLLECTION_NAME).document(binId)
                    .delete().get().getUpdateTime().toString();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot delete bin: operation interrupted", e);
        } catch (ExecutionException e) {
            throw new ServiceException("Cannot delete bin: " + binId, e.getCause());
        }
    }
}

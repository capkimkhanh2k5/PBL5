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
import com.google.cloud.firestore.FieldValue;
import com.google.cloud.firestore.SetOptions;
import java.util.HashMap;
import java.util.Map;

@Service
@RequiredArgsConstructor
public class BinMetadataService {

    private static final String COLLECTION_NAME = "bins_metadata";

    private static final String COMMAND_COLLECTION_NAME = "bin_commands";

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

            if (bin.getClassificationEnabled() == null) {
                bin.setClassificationEnabled(true);
            }

            if (bin.getClassificationUpdatedAt() == null) {
                bin.setClassificationUpdatedAt(Timestamp.now());
            }

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
            DocumentReference docRef =
                    firestore.collection(COLLECTION_NAME).document(binId);

            boolean exists = docRef.get().get().exists();

            Map<String, Object> updates = new HashMap<>();

            if (bin.getName() != null) {
                updates.put("name", bin.getName());
            }

            if (bin.getLocationDescription() != null) {
                updates.put("location_description", bin.getLocationDescription());
            }

            if (bin.getLatitude() != null) {
                updates.put("latitude", bin.getLatitude());
            }

            if (bin.getLongitude() != null) {
                updates.put("longitude", bin.getLongitude());
            }

            if (!exists) {
                updates.put("installed_at", Timestamp.now());
                updates.put("classification_enabled", true);
                updates.put("classification_updated_at", Timestamp.now());
            }

            return docRef
                    .set(updates, SetOptions.merge())
                    .get()
                    .getUpdateTime()
                    .toString();

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot update bin: operation interrupted", e);
        } catch (ExecutionException e) {
            throw new ServiceException("Cannot update bin: " + binId, e.getCause());
        }
    }

    public Map<String, Object> sendClassificationCommand(String binId, String rawValue) {
        try {
            if (binId == null || binId.trim().isEmpty()) {
                throw new ServiceException("binId must not be empty");
            }

            String value = rawValue == null ? "" : rawValue.trim().toUpperCase();

            if (!value.equals("ON") && !value.equals("OFF")) {
                throw new ServiceException("Command value must be ON or OFF");
            }

            DocumentReference binRef =
                    firestore.collection(COLLECTION_NAME).document(binId);

            if (!binRef.get().get().exists()) {
                throw new ResourceNotFoundException("Bin", binId);
            }

            String commandId = "cmd_" + System.currentTimeMillis();

            Map<String, Object> command = new HashMap<>();
            command.put("command_id", commandId);
            command.put("type", "CLASSIFICATION");
            command.put("value", value);
            command.put("status", "PENDING");
            command.put("created_at", FieldValue.serverTimestamp());
            command.put("handled_at", null);
            command.put("error_message", null);

            firestore
                    .collection(COMMAND_COLLECTION_NAME)
                    .document(binId)
                    .set(command, SetOptions.merge())
                    .get();

            Map<String, Object> response = new HashMap<>();
            response.put("binId", binId);
            response.put("commandId", commandId);
            response.put("type", "CLASSIFICATION");
            response.put("value", value);
            response.put("status", "PENDING");

            return response;

        } catch (ResourceNotFoundException e) {
            throw e;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot send classification command: operation interrupted", e);
        } catch (ExecutionException e) {
            throw new ServiceException("Cannot send classification command for bin: " + binId, e.getCause());
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

package com.iotSmartTrash.service;

import com.google.cloud.firestore.Firestore;
import com.google.cloud.firestore.QueryDocumentSnapshot;
import com.iotSmartTrash.exception.ServiceException;
import com.iotSmartTrash.model.TrashCategory;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

@Service
@RequiredArgsConstructor
public class TrashCategoryService {

    private static final String COLLECTION_NAME = "trash_categories";

    private final Firestore firestore;

    public List<TrashCategory> getAllCategories() {
        try {
            List<TrashCategory> categories = new ArrayList<>();
            for (QueryDocumentSnapshot doc : firestore.collection(COLLECTION_NAME).get().get().getDocuments()) {
                TrashCategory category = doc.toObject(TrashCategory.class);
                category.setId(doc.getId());
                categories.add(category);
            }
            return categories;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ServiceException("Cannot get list of trash categories: operation interrupted", e);
        } catch (ExecutionException e) {
            throw new ServiceException("Cannot get list of trash categories", e.getCause());
        }
    }
}

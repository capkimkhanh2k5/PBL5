package com.iotSmartTrash.service;

import com.google.cloud.firestore.Firestore;
import com.google.cloud.firestore.QueryDocumentSnapshot;
import com.google.firebase.cloud.FirestoreClient;
import com.iotSmartTrash.model.TrashCategory;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

@Service
public class TrashCategoryService {

    private static final String COLLECTION_NAME = "trash_categories";

    public List<TrashCategory> getAllCategories() throws ExecutionException, InterruptedException {
        Firestore dbFirestore = FirestoreClient.getFirestore();
        List<TrashCategory> categories = new ArrayList<>();

        // Truy vấn toàn bộ document trong collection 'trash_categories'
        for (QueryDocumentSnapshot document : dbFirestore.collection(COLLECTION_NAME).get().get().getDocuments()) {
            TrashCategory category = document.toObject(TrashCategory.class);
            category.setId(document.getId());
            categories.add(category);
        }

        return categories;
    }
}

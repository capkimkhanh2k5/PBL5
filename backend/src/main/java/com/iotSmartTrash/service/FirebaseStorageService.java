package com.iotSmartTrash.service;

import com.google.cloud.storage.Bucket;
import com.google.firebase.cloud.StorageClient;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.UUID;

@Service
public class FirebaseStorageService {

    public String uploadImage(MultipartFile file) throws IOException {
        String fileName = generateFileName(file.getOriginalFilename());
        Bucket bucket = StorageClient.getInstance().bucket();

        // Upload the file to Firebase Storage
        bucket.create(
                fileName,
                file.getInputStream(),
                file.getContentType());

        // Firebase Storage URLs
        String downloadUrl = String.format("https://firebasestorage.googleapis.com/v0/b/%s/o/%s?alt=media",
                bucket.getName(),
                URLEncoder.encode(fileName, StandardCharsets.UTF_8.toString()));

        return downloadUrl;
    }

    private String generateFileName(String originalFileName) {
        String extension = "";
        if (originalFileName != null && originalFileName.contains(".")) {
            extension = originalFileName.substring(originalFileName.lastIndexOf("."));
        }
        return UUID.randomUUID().toString() + extension;
    }
}

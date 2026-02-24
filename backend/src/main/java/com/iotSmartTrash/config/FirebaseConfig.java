package com.iotSmartTrash.config;

import com.google.auth.oauth2.GoogleCredentials;
import com.google.cloud.firestore.Firestore;
import com.google.firebase.FirebaseApp;
import com.google.firebase.FirebaseOptions;
import com.google.firebase.cloud.FirestoreClient;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import jakarta.annotation.PostConstruct;
import java.io.InputStream;

@Configuration
public class FirebaseConfig {

    @Value("${firebase.service-account:classpath:serviceAccountKey.json}")
    private String serviceAccountPath;

    @Value("${firebase.database-url}")
    private String databaseUrl;

    @Value("${firebase.storage-bucket}")
    private String storageBucket;

    @PostConstruct
    public void initialize() {
        try {
            InputStream serviceAccount = resolveServiceAccount();

            if (serviceAccount == null) {
                System.err.println("Not found file serviceAccountKey.json");
                return;
            }

            FirebaseOptions options = FirebaseOptions.builder()
                    .setCredentials(GoogleCredentials.fromStream(serviceAccount))
                    .setDatabaseUrl(databaseUrl)
                    .setStorageBucket(storageBucket)
                    .build();

            if (FirebaseApp.getApps().isEmpty()) {
                FirebaseApp.initializeApp(options);
                System.out.println("Firebase Admin SDK generated completed!");
            }
        } catch (Exception e) {
            System.err.println("Firebase init error: " + e.getMessage());
            e.printStackTrace();
        }
    }

    @Bean
    public Firestore firestore() {
        return FirestoreClient.getFirestore();
    }

    private InputStream resolveServiceAccount() {
        if (serviceAccountPath.startsWith("file:")) {
            String filePath = serviceAccountPath.substring(5);
            try {
                return new java.io.FileInputStream(filePath);
            } catch (java.io.FileNotFoundException e) {
                System.err.println("Not found file: " + filePath);
                return null;
            }
        } else if (serviceAccountPath.startsWith("classpath:")) {
            String resourceName = serviceAccountPath.substring(10);
            return getClass().getClassLoader().getResourceAsStream(resourceName);
        }
        return getClass().getClassLoader().getResourceAsStream("serviceAccountKey.json");
    }
}

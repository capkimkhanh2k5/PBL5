package com.iotSmartTrash.config;

import com.google.auth.oauth2.GoogleCredentials;
import com.google.firebase.FirebaseApp;
import com.google.firebase.FirebaseOptions;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;

import jakarta.annotation.PostConstruct;
import java.io.InputStream;

@Configuration
public class FirebaseConfig {

    @Value("${firebase.service-account:classpath:serviceAccountKey.json}")
    private String serviceAccountPath;

    @PostConstruct
    public void initialize() {
        try {
            InputStream serviceAccount;

            if (serviceAccountPath.startsWith("file:")) {
                // Docker: read from mounted file path
                String filePath = serviceAccountPath.substring(5);
                serviceAccount = new java.io.FileInputStream(filePath);
            } else if (serviceAccountPath.startsWith("classpath:")) {
                // Local: read from classpath resources
                String resourceName = serviceAccountPath.substring(10);
                serviceAccount = getClass().getClassLoader().getResourceAsStream(resourceName);
            } else {
                serviceAccount = getClass().getClassLoader().getResourceAsStream("serviceAccountKey.json");
            }

            if (serviceAccount == null) {
                System.err.println(" Không tìm thấy file serviceAccountKey.json");
                return;
            }

            FirebaseOptions options = FirebaseOptions.builder()
                    .setCredentials(GoogleCredentials.fromStream(serviceAccount))
                    .setDatabaseUrl("https://pbl5-f21e6-default-rtdb.asia-southeast1.firebasedatabase.app")
                    .build();

            if (FirebaseApp.getApps().isEmpty()) {
                FirebaseApp.initializeApp(options);
                System.out.println("Firebase Admin SDK khởi tạo thành công");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

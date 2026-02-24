package com.iotSmartTrash.model;

import com.google.cloud.Timestamp;
import com.google.cloud.firestore.annotation.PropertyName;
import com.iotSmartTrash.model.enums.UserRole;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class User {
    private String uid;
    private String username;
    private String email;
    private UserRole role;

    @PropertyName("avatar_url")
    private String avatarUrl;

    @PropertyName("created_at")
    private Timestamp createdAt;
}

package com.iotSmartTrash.model;

import com.google.cloud.Timestamp;
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
    private String role;
    private String avatar_url;
    private Timestamp created_at;
}

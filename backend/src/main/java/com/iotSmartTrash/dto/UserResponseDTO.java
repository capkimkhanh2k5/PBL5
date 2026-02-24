package com.iotSmartTrash.dto;

import com.google.cloud.Timestamp;
import com.iotSmartTrash.model.User;
import com.iotSmartTrash.model.enums.UserRole;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class UserResponseDTO {
    private String uid;
    private String username;
    private String email;
    private UserRole role;
    private String avatarUrl;
    private Timestamp createdAt;

    public static UserResponseDTO fromModel(User user) {
        if (user == null)
            return null;
        return UserResponseDTO.builder()
                .uid(user.getUid())
                .username(user.getUsername())
                .email(user.getEmail())
                .role(user.getRole())
                .avatarUrl(user.getAvatarUrl())
                .createdAt(user.getCreatedAt())
                .build();
    }
}

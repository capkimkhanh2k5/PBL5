package com.iotSmartTrash.dto;

import com.iotSmartTrash.model.enums.UserRole;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class UserRoleUpdateDTO {
    @NotNull(message = "Role is required")
    private UserRole role;
}

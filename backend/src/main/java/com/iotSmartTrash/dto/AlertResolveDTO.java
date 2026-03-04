package com.iotSmartTrash.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class AlertResolveDTO {
    @NotBlank(message = "Resolver ID is required")
    private String resolvedBy;
}

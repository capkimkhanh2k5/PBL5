package com.iotSmartTrash.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class ClassificationCommandRequest {

    @NotBlank(message = "Command value must not be blank")
    private String value;
}
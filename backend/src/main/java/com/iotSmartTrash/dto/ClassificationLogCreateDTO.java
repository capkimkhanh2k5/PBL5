package com.iotSmartTrash.dto;

import com.iotSmartTrash.model.ClassificationLog;
import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ClassificationLogCreateDTO {
    @NotBlank(message = "Bin ID is required")
    private String binId;

    @NotBlank(message = "Classification result is required")
    private String classificationResult;

    @NotNull(message = "Confidence score is required")
    @DecimalMin(value = "0.0", message = "Confidence score must be >= 0")
    @DecimalMax(value = "1.0", message = "Confidence score must be <= 1")
    private Double confidenceScore;

    public ClassificationLog toModel() {
        return ClassificationLog.builder()
                .binId(this.binId)
                .classificationResult(this.classificationResult)
                .confidenceScore(this.confidenceScore)
                .build();
    }
}

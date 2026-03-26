package com.iotSmartTrash.dto;

import com.iotSmartTrash.model.ClassificationLog;
import jakarta.validation.constraints.NotBlank;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ClassificationLogCreateDTO {
    @NotBlank(message = "Bin ID is required")
    private String binId;

    @NotBlank(message = "Image URL is required")
    private String imageUrl;

    @NotBlank(message = "Classification result is required")
    private String classificationResult;

    public ClassificationLog toModel() {
        return ClassificationLog.builder()
                .binId(this.binId)
                .imageUrl(this.imageUrl)
                .classificationResult(this.classificationResult)
                .build();
    }
}

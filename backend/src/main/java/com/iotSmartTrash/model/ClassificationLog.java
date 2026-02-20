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
public class ClassificationLog {
    private String log_id;
    private String bin_id;
    private String image_url;
    private String classification_result;
    private Double confidence_score;
    private Timestamp classified_at;
}

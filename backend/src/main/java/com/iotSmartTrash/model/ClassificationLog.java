package com.iotSmartTrash.model;

import com.google.cloud.Timestamp;
import com.google.cloud.firestore.annotation.PropertyName;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ClassificationLog {
    @PropertyName("log_id")
    private String logId;

    @PropertyName("bin_id")
    private String binId;

    @PropertyName("image_url")
    private String imageUrl;

    @PropertyName("classification_result")
    private String classificationResult;

    @PropertyName("confidence_score")
    private Double confidenceScore;

    @PropertyName("classified_at")
    private Timestamp classifiedAt;
}

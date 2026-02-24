package com.iotSmartTrash.model;

import com.google.cloud.Timestamp;
import com.google.cloud.firestore.annotation.PropertyName;
import com.iotSmartTrash.model.enums.AlertSeverity;
import com.iotSmartTrash.model.enums.AlertStatus;
import com.iotSmartTrash.model.enums.AlertType;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Alert {
    private String id;

    @PropertyName("bin_id")
    private String binId;

    @PropertyName("alert_type")
    private AlertType alertType;

    private AlertSeverity severity;
    private String message;
    private AlertStatus status;

    @PropertyName("resolved_by")
    private String resolvedBy;

    @PropertyName("created_at")
    private Timestamp createdAt;

    @PropertyName("resolved_at")
    private Timestamp resolvedAt;
}

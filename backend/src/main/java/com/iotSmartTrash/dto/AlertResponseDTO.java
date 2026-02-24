package com.iotSmartTrash.dto;

import com.google.cloud.Timestamp;
import com.iotSmartTrash.model.Alert;
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
public class AlertResponseDTO {
    private String id;
    private String binId;
    private AlertType alertType;
    private AlertSeverity severity;
    private String message;
    private AlertStatus status;
    private String resolvedBy;
    private Timestamp createdAt;
    private Timestamp resolvedAt;

    public static AlertResponseDTO fromModel(Alert alert) {
        if (alert == null)
            return null;
        return AlertResponseDTO.builder()
                .id(alert.getId())
                .binId(alert.getBinId())
                .alertType(alert.getAlertType())
                .severity(alert.getSeverity())
                .message(alert.getMessage())
                .status(alert.getStatus())
                .resolvedBy(alert.getResolvedBy())
                .createdAt(alert.getCreatedAt())
                .resolvedAt(alert.getResolvedAt())
                .build();
    }
}

package com.iotSmartTrash.dto;

import com.iotSmartTrash.model.Alert;
import com.iotSmartTrash.model.enums.AlertSeverity;
import com.iotSmartTrash.model.enums.AlertStatus;
import com.iotSmartTrash.model.enums.AlertType;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class AlertCreateDTO {
    @NotBlank(message = "Bin ID is required")
    private String binId;

    @NotNull(message = "Alert type is required")
    private AlertType alertType;

    @NotNull(message = "Severity is required")
    private AlertSeverity severity;

    private String message;

    public Alert toModel() {
        return Alert.builder()
                .binId(this.binId)
                .alertType(this.alertType)
                .severity(this.severity)
                .message(this.message)
                .status(AlertStatus.NEW)
                .build();
    }
}

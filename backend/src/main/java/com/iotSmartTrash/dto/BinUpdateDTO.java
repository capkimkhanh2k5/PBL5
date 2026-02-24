package com.iotSmartTrash.dto;

import com.iotSmartTrash.model.BinMetadata;
import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * DTO riêng cho PUT update bin — tất cả fields đều optional.
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class BinUpdateDTO {
    private String name;
    private String locationDescription;

    @DecimalMin(value = "-90.0", message = "Latitude must be >= -90")
    @DecimalMax(value = "90.0", message = "Latitude must be <= 90")
    private Double latitude;

    @DecimalMin(value = "-180.0", message = "Longitude must be >= -180")
    @DecimalMax(value = "180.0", message = "Longitude must be <= 180")
    private Double longitude;

    public BinMetadata toModel() {
        return BinMetadata.builder()
                .name(this.name)
                .locationDescription(this.locationDescription)
                .latitude(this.latitude)
                .longitude(this.longitude)
                .build();
    }
}

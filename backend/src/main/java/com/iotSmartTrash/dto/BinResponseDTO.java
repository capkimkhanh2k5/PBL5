package com.iotSmartTrash.dto;

import com.google.cloud.Timestamp;
import com.iotSmartTrash.model.BinMetadata;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class BinResponseDTO {
    private String id;
    private String name;
    private String locationDescription;
    private Double latitude;
    private Double longitude;
    private Timestamp installedAt;

    public static BinResponseDTO fromModel(BinMetadata bin) {
        if (bin == null)
            return null;
        return BinResponseDTO.builder()
                .id(bin.getId())
                .name(bin.getName())
                .locationDescription(bin.getLocationDescription())
                .latitude(bin.getLatitude())
                .longitude(bin.getLongitude())
                .installedAt(bin.getInstalledAt())
                .build();
    }
}

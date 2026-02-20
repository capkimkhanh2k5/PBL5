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
public class BinMetadata {
    private String id;
    private String name;
    private String location_description;
    private Double latitude;
    private Double longitude;
    private Timestamp installed_at;
}

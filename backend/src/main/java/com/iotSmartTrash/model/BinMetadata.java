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
public class BinMetadata {
    private String id;
    private String name;

    @PropertyName("location_description")
    private String locationDescription;

    private Double latitude;
    private Double longitude;

    @PropertyName("installed_at")
    private Timestamp installedAt;
}

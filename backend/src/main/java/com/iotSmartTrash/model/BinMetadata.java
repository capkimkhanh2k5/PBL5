package com.iotSmartTrash.model;

import com.google.cloud.Timestamp;
import com.google.cloud.firestore.annotation.PropertyName;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/*
    Model Meta Data của Bin
    Lưu vị trí, địa chỉ, mô tả
*/
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class BinMetadata {
    private String id;
    private String name;

    private String locationDescription;

    private Double latitude;
    private Double longitude;

    private Timestamp installedAt;
    private Boolean classificationEnabled;

    private Timestamp classificationUpdatedAt;

    @PropertyName("location_description")
    public String getLocationDescription() {
        return locationDescription;
    }

    @PropertyName("location_description")
    public void setLocationDescription(String locationDescription) {
        this.locationDescription = locationDescription;
    }

    @PropertyName("installed_at")
    public Timestamp getInstalledAt() {
        return installedAt;
    }

    @PropertyName("installed_at")
    public void setInstalledAt(Timestamp installedAt) {
        this.installedAt = installedAt;
    }

    @PropertyName("classification_enabled")
    public Boolean getClassificationEnabled() {
        return classificationEnabled;
    }

    @PropertyName("classification_enabled")
    public void setClassificationEnabled(Boolean classificationEnabled) {
        this.classificationEnabled = classificationEnabled;
    }

    @PropertyName("classification_updated_at")
    public Timestamp getClassificationUpdatedAt() {
        return classificationUpdatedAt;
    }

    @PropertyName("classification_updated_at")
    public void setClassificationUpdatedAt(Timestamp classificationUpdatedAt) {
        this.classificationUpdatedAt = classificationUpdatedAt;
    }
}
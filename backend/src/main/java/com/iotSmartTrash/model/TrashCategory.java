package com.iotSmartTrash.model;

import com.google.cloud.firestore.annotation.PropertyName;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class TrashCategory {
    private String id;
    private String name;
    private String type;

    @PropertyName("icon_url")
    private String iconUrl;

    private String description;
}

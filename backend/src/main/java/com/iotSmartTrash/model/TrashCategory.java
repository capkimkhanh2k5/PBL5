package com.iotSmartTrash.model;

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
    private String icon_url;
    private String description;
}

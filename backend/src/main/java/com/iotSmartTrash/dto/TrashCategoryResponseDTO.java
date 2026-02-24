package com.iotSmartTrash.dto;

import com.iotSmartTrash.model.TrashCategory;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class TrashCategoryResponseDTO {
    private String id;
    private String name;
    private String type;
    private String iconUrl;
    private String description;

    public static TrashCategoryResponseDTO fromModel(TrashCategory category) {
        if (category == null)
            return null;
        return TrashCategoryResponseDTO.builder()
                .id(category.getId())
                .name(category.getName())
                .type(category.getType())
                .iconUrl(category.getIconUrl())
                .description(category.getDescription())
                .build();
    }
}

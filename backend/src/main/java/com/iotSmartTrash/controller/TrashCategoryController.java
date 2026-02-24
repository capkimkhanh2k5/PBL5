package com.iotSmartTrash.controller;

import com.iotSmartTrash.dto.TrashCategoryResponseDTO;
import com.iotSmartTrash.model.TrashCategory;
import com.iotSmartTrash.service.TrashCategoryService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/v1/categories")
@RequiredArgsConstructor
public class TrashCategoryController {

    private final TrashCategoryService trashCategoryService;

    @GetMapping
    public ResponseEntity<List<TrashCategoryResponseDTO>> getAllCategories() {
        List<TrashCategory> categories = trashCategoryService.getAllCategories();
        List<TrashCategoryResponseDTO> dtos = categories.stream()
                .map(TrashCategoryResponseDTO::fromModel)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
}

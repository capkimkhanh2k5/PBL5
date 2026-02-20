package com.iotSmartTrash.controller;

import com.iotSmartTrash.model.TrashCategory;
import com.iotSmartTrash.service.TrashCategoryService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.concurrent.ExecutionException;

@RestController
@RequestMapping("/api/v1/categories")
@RequiredArgsConstructor
public class TrashCategoryController {

    private final TrashCategoryService trashCategoryService;

    @GetMapping
    public ResponseEntity<List<TrashCategory>> getAllCategories() throws ExecutionException, InterruptedException {
        List<TrashCategory> categories = trashCategoryService.getAllCategories();
        return ResponseEntity.ok(categories);
    }
}

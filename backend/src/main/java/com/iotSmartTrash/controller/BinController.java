package com.iotSmartTrash.controller;

import com.iotSmartTrash.dto.BinCreateDTO;
import com.iotSmartTrash.dto.BinResponseDTO;
import com.iotSmartTrash.dto.BinUpdateDTO;
import com.iotSmartTrash.model.BinMetadata;
import com.iotSmartTrash.service.BinMetadataService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.stream.Collectors;
import com.iotSmartTrash.dto.ClassificationCommandRequest;
import java.util.Map;

@RestController
@RequestMapping("/api/v1/bins")
@RequiredArgsConstructor
@Validated
public class BinController {

    private final BinMetadataService binService;

    @GetMapping
    public ResponseEntity<List<BinResponseDTO>> getAllBins() {
        List<BinMetadata> bins = binService.getAllBins();
        List<BinResponseDTO> dtos = bins.stream()
                .map(BinResponseDTO::fromModel)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }

    @GetMapping("/{id}")
    public ResponseEntity<BinResponseDTO> getBinById(@PathVariable String id) {
        BinMetadata bin = binService.getBinById(id);
        return ResponseEntity.ok(BinResponseDTO.fromModel(bin));
        // ResourceNotFoundException → GlobalExceptionHandler → 404
    }

    @PostMapping
    public ResponseEntity<String> createBin(@Valid @RequestBody BinCreateDTO binDto) {
        BinMetadata bin = binDto.toModel();
        String updateTime = binService.createBin(bin);
        return ResponseEntity.ok("Successfully created bin at " + updateTime);
    }

    @PutMapping("/{id}")
    public ResponseEntity<String> updateBin(
            @PathVariable String id,
            @Valid @RequestBody BinUpdateDTO binDto) {
        BinMetadata bin = binDto.toModel();
        String updateTime = binService.updateBin(id, bin);
        return ResponseEntity.ok("Successfully updated bin at " + updateTime);
    }

    @PatchMapping("/{id}/classification-command")
    public ResponseEntity<Map<String, Object>> sendClassificationCommand(
            @PathVariable String id,
            @Valid @RequestBody ClassificationCommandRequest request) {

        Map<String, Object> response =
                binService.sendClassificationCommand(id, request.getValue());

        return ResponseEntity.ok(response);
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<String> deleteBin(@PathVariable String id) {
        String updateTime = binService.deleteBin(id);
        return ResponseEntity.ok("Successfully deleted bin at " + updateTime);
    }
}

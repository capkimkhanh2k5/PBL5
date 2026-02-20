package com.iotSmartTrash.controller;

import com.iotSmartTrash.model.BinMetadata;
import com.iotSmartTrash.service.BinMetadataService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.concurrent.ExecutionException;

@RestController
@RequestMapping("/api/v1/bins")
@RequiredArgsConstructor
public class BinController {

    private final BinMetadataService binService;

    @GetMapping
    public ResponseEntity<List<BinMetadata>> getAllBins() throws ExecutionException, InterruptedException {
        return ResponseEntity.ok(binService.getAllBins());
    }

    @GetMapping("/{id}")
    public ResponseEntity<BinMetadata> getBinById(@PathVariable String id)
            throws ExecutionException, InterruptedException {
        BinMetadata bin = binService.getBinById(id);
        if (bin != null) {
            return ResponseEntity.ok(bin);
        }
        return ResponseEntity.notFound().build();
    }

    @PostMapping
    public ResponseEntity<String> createBin(@RequestBody BinMetadata bin)
            throws ExecutionException, InterruptedException {
        String updateTime = binService.createBin(bin);
        return ResponseEntity.ok("Successfully created bin at " + updateTime);
    }

    @PutMapping("/{id}")
    public ResponseEntity<String> updateBin(@PathVariable String id, @RequestBody BinMetadata bin)
            throws ExecutionException, InterruptedException {
        String updateTime = binService.updateBin(id, bin);
        return ResponseEntity.ok("Successfully updated bin at " + updateTime);
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<String> deleteBin(@PathVariable String id) throws ExecutionException, InterruptedException {
        String updateTime = binService.deleteBin(id);
        return ResponseEntity.ok("Successfully deleted bin at " + updateTime);
    }
}

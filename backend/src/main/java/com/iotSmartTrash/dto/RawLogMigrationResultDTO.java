package com.iotSmartTrash.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class RawLogMigrationResultDTO {
    private int binsScanned;
    private int docsScanned;
    private int docsUpdated;
    private int docsTimestampBackfilled;
}

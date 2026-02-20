package com.iotSmartTrash;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

@SpringBootApplication
@EnableScheduling
public class SmartTrashBackendApplication {

    public static void main(String[] args) {
        SpringApplication.run(SmartTrashBackendApplication.class, args);
    }

}

package com.iotSmartTrash.config;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.lang.NonNull;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.util.Set;

/**
 * Filter xác thực API Key cho các IoT device endpoints.
 * Header cần có: X-IoT-API-Key: <configured-key>
 */
@Component
public class IoTApiKeyFilter extends OncePerRequestFilter {

    private static final Set<String> PROTECTED_IOT_PATHS = Set.of(
            "/api/v1/system/classification-logs",
            "/api/v1/system/alerts");

    @Value("${iot.api-key}")
    private String validApiKey;

    @Override
    protected void doFilterInternal(@NonNull HttpServletRequest request,
            @NonNull HttpServletResponse response,
            @NonNull FilterChain filterChain) throws ServletException, IOException {

        String path = request.getRequestURI();
        String method = request.getMethod();

        // Chỉ bảo vệ POST requests đến IoT endpoints
        boolean isIoTEndpoint = PROTECTED_IOT_PATHS.stream().anyMatch(path::startsWith)
                && "POST".equalsIgnoreCase(method);

        if (!isIoTEndpoint) {
            filterChain.doFilter(request, response);
            return;
        }

        String providedKey = request.getHeader("X-IoT-API-Key");
        if (providedKey == null || !providedKey.equals(validApiKey)) {
            response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
            response.setContentType("application/json");
            response.setCharacterEncoding("UTF-8");
            response.getWriter().write("""
                    {"error": "INVALID_IOT_API_KEY", "message": "X-IoT-API-Key header is invalid or missing"}
                    """);
            return;
        }

        filterChain.doFilter(request, response);
    }
}

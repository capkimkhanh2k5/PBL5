package com.iotSmartTrash.config;

import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.CorsConfigurationSource;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;

import java.util.Arrays;

@Configuration
@EnableWebSecurity
@RequiredArgsConstructor
public class SecurityConfig {

    private final FirebaseTokenFilter firebaseTokenFilter;

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
                // Tắt CSRF, bật CORS vì chương trình sử dụng JWT Token (Bearer Token)
                .csrf(csrf -> csrf.disable())
                .cors(cors -> cors.configurationSource(corsConfigurationSource()))

                // Không lưu session trên Session Cookie của Server
                .sessionManagement(session -> session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))

                // Quy tắc Permission cho API
                .authorizeHttpRequests(auth -> auth
                        // Cho phép gọi thẳng không cần Token
                        .requestMatchers("/api/v1/health").permitAll()
                        .requestMatchers("/api/v1/trigger/**").permitAll() // Test CronJob Public
                        .requestMatchers("/api/v1/system/classification-logs").permitAll() // Cho IoT Pi Push Ảnh
                        .requestMatchers("/api/v1/system/alerts").permitAll() // Cho IoT Pi Tạo Cảnh báo
                        .requestMatchers("/swagger-ui/**", "/v3/api-docs/**").permitAll() // Mở Swagger UI để Test

                        // Bắt buộc tất cả các API còn lại phải có Bearer Token
                        .anyRequest().authenticated())
                // Chèn Gác cổng Firebase của chúng ta vào đầu bảo vệ
                .addFilterBefore(firebaseTokenFilter, UsernamePasswordAuthenticationFilter.class);

        return http.build();
    }

    /**
     * Cấu hình CORS hoàn chỉnh cho phép các giao diện Web/Mobile bên ngoài gọi vào
     * Backend
     * Tránh lỗi "No Access-Control-Allow-Origin header is present" trên Browser.
     */
    @Bean
    public CorsConfigurationSource corsConfigurationSource() {
        CorsConfiguration configuration = new CorsConfiguration();

        // Cho phép các tên miền Frontend
        configuration.setAllowedOrigins(
                Arrays.asList("http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"));

        configuration.setAllowedMethods(Arrays.asList("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"));

        // Cho phép gửi kèm các Type Header thông dụng
        configuration.setAllowedHeaders(Arrays.asList("Authorization", "Content-Type", "Cache-Control"));

        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        // Áp dụng bộ luật CORS trên cho TOÀN BỘ các đường dẫn API ("/**")
        source.registerCorsConfiguration("/**", configuration);
        return source;
    }
}

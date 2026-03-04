package com.iotSmartTrash.config;

import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
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
import java.util.List;

@Configuration
@EnableWebSecurity
@RequiredArgsConstructor
public class SecurityConfig {

        private final FirebaseTokenFilter firebaseTokenFilter;
        private final IoTApiKeyFilter ioTApiKeyFilter;

        /**
         * Read from application.yml — override with env CORS_ALLOWED_ORIGINS on deploy
         */
        @Value("${cors.allowed-origins}")
        private List<String> allowedOrigins;

        @Bean
        public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
                http
                                .csrf(csrf -> csrf.disable())
                                .cors(cors -> cors.configurationSource(corsConfigurationSource()))
                                .sessionManagement(session -> session
                                                .sessionCreationPolicy(SessionCreationPolicy.STATELESS))
                                .authorizeHttpRequests(auth -> auth
                                                // Public — no token required
                                                .requestMatchers("/api/v1/health").permitAll()
                                                .requestMatchers("/swagger-ui/**", "/v3/api-docs/**").permitAll()

                                                // IoT endpoints — protected by IoTApiKeyFilter (X-IoT-API-Key header)
                                                .requestMatchers("/api/v1/system/classification-logs").permitAll()
                                                .requestMatchers("/api/v1/system/alerts").permitAll()

                                                // Dev/test only — disable or protect before production deploy
                                                .requestMatchers("/api/v1/trigger/**").permitAll()

                                                // All other requests require Firebase Bearer Token
                                                .anyRequest().authenticated())

                                // IoTApiKeyFilter runs BEFORE FirebaseTokenFilter
                                .addFilterBefore(ioTApiKeyFilter, UsernamePasswordAuthenticationFilter.class)
                                .addFilterBefore(firebaseTokenFilter, UsernamePasswordAuthenticationFilter.class);

                return http.build();
        }

        @Bean
        public CorsConfigurationSource corsConfigurationSource() {
                CorsConfiguration configuration = new CorsConfiguration();

                boolean hasWildcard = allowedOrigins != null
                                && allowedOrigins.stream().anyMatch("*"::equals);

                configuration.setAllowedOrigins(allowedOrigins);
                configuration.setAllowedMethods(Arrays.asList("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"));
                configuration.setAllowedHeaders(
                                Arrays.asList("Authorization", "Content-Type", "Cache-Control", "X-IoT-API-Key"));
                configuration.setAllowCredentials(!hasWildcard);

                UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
                source.registerCorsConfiguration("/**", configuration);
                return source;
        }
}

# ============================================
# Stage 1: Build with Maven
# ============================================
FROM maven:3.9-eclipse-temurin-21 AS build
WORKDIR /app

# Cache dependencies first
COPY pom.xml .
RUN mvn dependency:go-offline -B

# Copy source and build
COPY src ./src
RUN mvn package -DskipTests -B

# ============================================
# Stage 2: Runtime (Ubuntu Jammy - supports ARM64/Apple Silicon)
# ============================================
FROM eclipse-temurin:21-jre-jammy

# Security: non-root user
RUN groupadd -g 1001 appgroup && \
    useradd -u 1001 -g appgroup -s /bin/false appuser

WORKDIR /app

# Copy JAR from build stage
COPY --from=build --chown=appuser:appgroup /app/target/*.jar app.jar

# Create config directory for mounted secrets
RUN mkdir -p /app/config && chown appuser:appgroup /app/config

USER 1001
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/health || exit 1

ENTRYPOINT ["java", "-jar", "app.jar"]

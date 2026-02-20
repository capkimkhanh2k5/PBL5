package com.iotSmartTrash.config;

import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseAuthException;
import com.google.firebase.auth.FirebaseToken;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.lang.NonNull;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.util.ArrayList;

@Component
public class FirebaseTokenFilter extends OncePerRequestFilter {

    @Override
    protected void doFilterInternal(@NonNull HttpServletRequest request,
            @NonNull HttpServletResponse response,
            @NonNull FilterChain filterChain) throws ServletException, IOException {

        // Lấy mã Token từ Header "Authorization"
        String header = request.getHeader("Authorization");

        // Kiểm tra nếu không có mã Token, hoặc mã không bắt đầu bằng "Bearer "
        if (header == null || !header.startsWith("Bearer ")) {
            filterChain.doFilter(request, response);
            return;
        }

        // Cắt bỏ chữ "Bearer " để lấy đúng chuỗi JWT
        String token = header.substring(7);

        try {
            // Liên hệ Google (Firebase) để Verify xem Token này có Hợp Lệ không
            FirebaseToken decodedToken = FirebaseAuth.getInstance().verifyIdToken(token);

            // Nếu hợp lệ, lấy Username (hoặc Email/UID) ra
            String uid = decodedToken.getUid();

            // Nhét Bearer Token vào Spring Security
            UsernamePasswordAuthenticationToken authentication = new UsernamePasswordAuthenticationToken(
                    uid, null, new ArrayList<>());
            authentication.setDetails(new WebAuthenticationDetailsSource().buildDetails(request));

            // Đặt người dùng vào Context của request hiện tại
            SecurityContextHolder.getContext().setAuthentication(authentication);

        } catch (FirebaseAuthException e) {
            // Nếu token giả mạo, hết hạn -> Bị bắt ở đây
            System.err.println("Firebase Auth Error: " + e.getMessage());
            response.setStatus(HttpServletResponse.SC_UNAUTHORIZED); // Ném lỗi 401
            response.getWriter().write("{\"error\": \"Invalid or Expired Firebase Token! 401 Unauthorized\"}");
            return;
        }

        // Đi tiếp xuống Controller hoặc Filter tiếp theo
        filterChain.doFilter(request, response);
    }
}

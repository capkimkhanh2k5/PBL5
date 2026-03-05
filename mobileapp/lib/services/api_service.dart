import 'package:dio/dio.dart';
import 'auth_service.dart';

class ApiService {
  late final Dio _dio;
  final AuthService _authService;

  // Thay đổi baseUrl khi deploy (dùng biến env hoặc config)
  static const String _baseUrl = 'http://10.0.2.2:8080/api/v1'; // Android emulator → localhost
  // Dùng 'http://localhost:8080/api/v1' cho iOS simulator
  // Dùng 'http://<your-ip>:8080/api/v1' cho device thật

  ApiService({required AuthService authService}) : _authService = authService {
    _dio = Dio(BaseOptions(
      baseUrl: _baseUrl,
      connectTimeout: const Duration(seconds: 10),
      receiveTimeout: const Duration(seconds: 10),
      headers: {'Content-Type': 'application/json'},
    ));

    // Interceptor tự động gắn Firebase ID Token vào mọi request
    _dio.interceptors.add(InterceptorsWrapper(
      onRequest: (options, handler) async {
        final token = await _authService.getIdToken();
        if (token != null) {
          options.headers['Authorization'] = 'Bearer $token';
        }
        handler.next(options);
      },
      onError: (error, handler) {
        // Log hoặc handle lỗi chung (401, 500, v.v.)
        handler.next(error);
      },
    ));
  }

  // Sync user lên backend sau khi đăng ký / đăng nhập
  Future<Map<String, dynamic>> syncUser() async {
    final response = await _dio.post('/users/sync');
    return response.data;
  }

  // Lấy thông tin user hiện tại
  Future<Map<String, dynamic>> getMe() async {
    final response = await _dio.get('/users/me');
    return response.data;
  }

  // Generic GET
  Future<Response> get(String path, {Map<String, dynamic>? queryParams}) {
    return _dio.get(path, queryParameters: queryParams);
  }

  // Generic POST 
  Future<Response> post(String path, {dynamic data}) {
    return _dio.post(path, data: data);
  }
}

import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart' show kIsWeb, defaultTargetPlatform, TargetPlatform;
import 'auth_service.dart';

class ApiService {
  late final Dio _dio;
  final AuthService _authService;
  static const String _apiBaseUrlOverride = String.fromEnvironment('API_BASE_URL');

  // Ưu tiên API_BASE_URL từ --dart-define để tránh hard-code theo máy.
  // Ví dụ: flutter run --dart-define=API_BASE_URL=http://192.168.1.10:8080/api/v1
  static String get _baseUrl {
    if (_apiBaseUrlOverride.trim().isNotEmpty) {
      return _apiBaseUrlOverride;
    }

    if (kIsWeb) {
      return 'http://localhost:8080/api/v1';
    }

    switch (defaultTargetPlatform) {
      case TargetPlatform.android:
        // Mobile thật/emulator: dùng LAN IP của máy chạy backend.
        return 'http://192.168.4.83:8081/api/v1';
      case TargetPlatform.iOS:
        // iPhone thật không truy cập được localhost của Mac, dùng LAN IP.
        return 'http://192.168.1.10:8080/api/v1';
      default:
        // Desktop/dev fallback
        return 'http://192.168.4.83:8081/api/v1';
    }
  }

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

  Future<List<Map<String, dynamic>>> getAllBins() async {
    final response = await _dio.get('/bins');
    final data = response.data;
    if (data is List) {
      return data.map((e) => Map<String, dynamic>.from(e as Map)).toList();
    }
    return const [];
  }

  Future<Map<String, dynamic>> getBinById(String binId) async {
    final response = await _dio.get('/bins/$binId');
    return Map<String, dynamic>.from(response.data as Map);
  }

  Future<Map<String, dynamic>> getBinCoordinates(String binId) async {
    final bin = await getBinById(binId);
    return {
      'id': bin['id'] ?? binId,
      'latitude': (bin['latitude'] as num?)?.toDouble(),
      'longitude': (bin['longitude'] as num?)?.toDouble(),
    };
  }

  Future<void> updateBinCoordinates(
    String binId, {
    required double latitude,
    required double longitude,
  }) async {
    if (latitude < -90 || latitude > 90) {
      throw ArgumentError.value(latitude, 'latitude', 'Latitude must be in [-90, 90]');
    }
    if (longitude < -180 || longitude > 180) {
      throw ArgumentError.value(longitude, 'longitude', 'Longitude must be in [-180, 180]');
    }

    await _dio.put('/bins/$binId', data: {
      'latitude': latitude,
      'longitude': longitude,
    });
  }

  Future<void> connectBinFromQr({
    required String binId,
    required String name,
    String? locationDescription,
    required double latitude,
    required double longitude,
  }) async {
    final trimmedId = binId.trim();
    final trimmedName = name.trim();
    if (trimmedId.isEmpty) {
      throw ArgumentError.value(binId, 'binId', 'binId must not be empty');
    }
    if (trimmedName.isEmpty) {
      throw ArgumentError.value(name, 'name', 'name must not be empty');
    }
    if (latitude < -90 || latitude > 90) {
      throw ArgumentError.value(latitude, 'latitude', 'Latitude must be in [-90, 90]');
    }
    if (longitude < -180 || longitude > 180) {
      throw ArgumentError.value(longitude, 'longitude', 'Longitude must be in [-180, 180]');
    }

    await _dio.put('/bins/$trimmedId', data: {
      'name': trimmedName,
      'locationDescription': locationDescription?.trim(),
      'latitude': latitude,
      'longitude': longitude,
    });
  }

  Future<List<Map<String, dynamic>>> getAllBinStatuses() async {
    final response = await _dio.get('/iot/bins/status');
    final data = response.data;
    if (data is List) {
      return data.map((e) => Map<String, dynamic>.from(e as Map)).toList();
    }
    return const [];
  }

  Future<Map<String, dynamic>> getBinStatus(String binId) async {
    final response = await _dio.get('/iot/bins/$binId/status');
    return Map<String, dynamic>.from(response.data as Map);
  }

  Future<List<Map<String, dynamic>>> getRecentSensorLogs(String binId, {int limit = 30}) async {
    final response = await _dio.get('/iot/bins/$binId/sensor-logs', queryParameters: {'limit': limit});
    final data = response.data;
    if (data is List) {
      return data.map((e) => Map<String, dynamic>.from(e as Map)).toList();
    }
    return const [];
  }

  Future<List<Map<String, dynamic>>> getClassificationLogs({String? binId, int limit = 20}) async {
    final response = await _dio.get('/system/classification-logs', queryParameters: {
      'limit': limit,
      if (binId != null && binId.isNotEmpty) 'binId': binId,
    });
    final data = response.data;
    if (data is List) {
      return data.map((e) => Map<String, dynamic>.from(e as Map)).toList();
    }
    return const [];
  }

  Future<List<Map<String, dynamic>>> getPickupSchedule({int limit = 40}) async {
    final response = await _dio.get('/system/pickup-schedule', queryParameters: {
      'limit': limit,
    });
    final data = response.data;
    if (data is List) {
      return data.map((e) => Map<String, dynamic>.from(e as Map)).toList();
    }
    return const [];
  }

  Future<void> triggerAggregateSensorLogs() async {
    await _dio.post('/trigger/aggregate-sensor-logs');
  }

  // Generic GET
  Future<Response> get(String path, {Map<String, dynamic>? queryParams}) {
    return _dio.get(path, queryParameters: queryParams);
  }

  // Generic POST 
  Future<Response> post(String path, {dynamic data}) {
    return _dio.post(path, data: data);
  }

  Future<Map<String, dynamic>> getMyProfile() async {
    final response = await _dio.get('/settings/me');
    return response.data;
  }

  Future<void> updateUsername(String username) async {
    await _dio.put(
      '/settings/username',
      data: {
        'username': username,
      },
    );
  }
}


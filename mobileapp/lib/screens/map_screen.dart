import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:latlong2/latlong.dart';
import 'package:dio/dio.dart';
import 'package:geolocator/geolocator.dart';
import 'dart:async';
import 'dart:math' as Math;

class MapScreen extends StatefulWidget {
  final String? initialBinId;
  const MapScreen({super.key, this.initialBinId});

  @override
  State<MapScreen> createState() => MapScreenState();
}

class MapScreenState extends State<MapScreen> with TickerProviderStateMixin {
  final MapController _mapController = MapController();
  final TextEditingController _searchController = TextEditingController();

  LatLng? _previousCenter;
  double? _previousZoom;

  bool _isZoomedToMarker = false;
  bool _isAnimating = false;
  bool _showBinList = false;

  String? _selectedBinId;
  String _searchText = '';
  LatLng? _currentLocation;
  List<LatLng> _routePoints = [];
  bool _isRouting = false;
  String? _routeInfoText;

  StreamSubscription<Position>? _positionSub;
  bool _isNavigating = false;
  int _nextStepIndex = 0;
  String? _nextInstructionText;
  List<Map<String, dynamic>> _routeSteps = [];
  double _currentBearing = 0;
  LatLng? _centerBeforeRoute;
  double? _zoomBeforeRoute;

  late Future<List<Map<String, dynamic>>> _binsFuture;

  static const Color primaryGreen = Color(0xFF1F7A3A);
  static const Color lightGreen = Color(0xFFEAF6ED);
  static const Color orange = Color(0xFFF29C38);
  static const Color red = Color(0xFFE53935);

  @override
  void initState() {
    super.initState();

    if (widget.initialBinId != null) {
      _selectedBinId = widget.initialBinId;
    }

    _binsFuture = fetchBins();
  }

  @override
  void dispose() {
    _positionSub?.cancel();
    _searchController.dispose();
    super.dispose();
  }

  double _toDouble(dynamic value) {
    if (value == null) return 0;
    if (value is int) return value.toDouble();
    if (value is double) return value;
    if (value is num) return value.toDouble();
    return double.tryParse(value.toString()) ?? 0;
  }

  int _toInt(dynamic value) {
    if (value == null) return 0;
    if (value is int) return value;
    if (value is double) return value.round();
    if (value is num) return value.round();
    return int.tryParse(value.toString()) ?? 0;
  }

  Future<List<Map<String, dynamic>>> fetchBins() async {
    final snapshot =
    await FirebaseFirestore.instance.collection('bins_metadata').get();

    return snapshot.docs.map((doc) {
      final data = doc.data();

      return {
        'id': doc.id,
        'name': data['name'] ?? 'Unknown Bin',
        'latitude': _toDouble(data['latitude']),
        'longitude': _toDouble(data['longitude']),
        'location': data['location_description'] ??
            data['location'] ??
            'Chưa có địa điểm',
        'fillLevel': _toInt(
          data['fill_level'] ??
              data['trash_level'] ??
              data['fillPercent'] ??
              data['percentage'],
        ),
      };
    }).where((bin) {
      return bin['latitude'] != 0 && bin['longitude'] != 0;
    }).toList();
  }

  void _refreshBins() {
    setState(() {
      _binsFuture = fetchBins();
    });
  }

  void _startNavigation() {
    if (_routePoints.isEmpty || _currentLocation == null) return;

    setState(() {
      _isNavigating = true;
    });

    final nextPoint = _findNextRoutePoint(_currentLocation!);

    if (nextPoint != null) {
      final bearing = _calculateBearing(_currentLocation!, nextPoint);

      _mapController.moveAndRotate(
        _currentLocation!,
        18,
        _toMapRotation(bearing),
      );
    } else {
      _mapController.move(
        _currentLocation!,
        18,
      );
    }

    _startNavigationTracking();
  }

  Widget _buildRoutePreviewPanel(Map<String, dynamic>? selectedBin) {
    if (_routePoints.isEmpty || _isNavigating) {
      return const SizedBox.shrink();
    }

    return Positioned(
      left: 14,
      right: 14,
      bottom: 86,
      child: Material(
        elevation: 12,
        shadowColor: Colors.black26,
        borderRadius: BorderRadius.circular(24),
        child: Container(
          padding: const EdgeInsets.fromLTRB(18, 12, 18, 16),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(24),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 42,
                height: 5,
                margin: const EdgeInsets.only(bottom: 14),
                decoration: BoxDecoration(
                  color: Colors.grey.shade300,
                  borderRadius: BorderRadius.circular(20),
                ),
              ),

              Row(
                children: [
                  Container(
                    width: 38,
                    height: 38,
                    decoration: BoxDecoration(
                      color: lightGreen,
                      borderRadius: BorderRadius.circular(13),
                    ),
                    child: const Icon(
                      Icons.directions_car,
                      color: primaryGreen,
                      size: 23,
                    ),
                  ),

                  const SizedBox(width: 12),

                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'Lái xe',
                          style: TextStyle(
                            fontSize: 19,
                            fontWeight: FontWeight.w900,
                          ),
                        ),
                        const SizedBox(height: 3),
                        Text(
                          _routeInfoText ?? '',
                          style: const TextStyle(
                            fontSize: 14,
                            color: primaryGreen,
                            fontWeight: FontWeight.w800,
                          ),
                        ),
                        const SizedBox(height: 2),
                        Text(
                          selectedBin?['name'] ?? 'Thùng rác đã chọn',
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(
                            fontSize: 12,
                            color: Colors.black54,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ),
                  ),

                  InkWell(
                    customBorder: const CircleBorder(),
                    onTap: _clearRoute,
                    child: Container(
                      width: 36,
                      height: 36,
                      decoration: BoxDecoration(
                        color: Colors.grey.shade100,
                        shape: BoxShape.circle,
                      ),
                      child: const Icon(
                        Icons.close,
                        size: 21,
                      ),
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 14),

              Row(
                children: [
                  Expanded(
                    child: SizedBox(
                      height: 46,
                      child: ElevatedButton.icon(
                        onPressed: _startNavigation,
                        icon: const Icon(
                          Icons.navigation,
                          size: 18,
                        ),
                        label: const Text('Bắt đầu'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: primaryGreen,
                          foregroundColor: Colors.white,
                          elevation: 0,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(14),
                          ),
                          textStyle: const TextStyle(
                            fontSize: 15,
                            fontWeight: FontWeight.w900,
                          ),
                        ),
                      ),
                    ),
                  ),

                  const SizedBox(width: 12),

                  Container(
                    width: 46,
                    height: 46,
                    decoration: BoxDecoration(
                      color: lightGreen,
                      borderRadius: BorderRadius.circular(14),
                    ),
                    child: IconButton(
                      onPressed: () {},
                      icon: const Icon(
                        Icons.ios_share,
                        color: primaryGreen,
                        size: 21,
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _clearRoute() {
    final restoreCenter = _centerBeforeRoute ?? _mapController.camera.center;
    final restoreZoom = _zoomBeforeRoute ?? _mapController.camera.zoom;

    setState(() {
      _routePoints.clear();
      _routeSteps.clear();
      _currentLocation = null;
      _routeInfoText = null;
      _nextInstructionText = null;
      _nextStepIndex = 0;
      _currentBearing = 0;
      _isNavigating = false;
    });

    _positionSub?.cancel();
    _positionSub = null;

    _mapController.moveAndRotate(
      restoreCenter,
      restoreZoom,
      0,
    );

    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;

      _mapController.moveAndRotate(
        restoreCenter,
        restoreZoom,
        0,
      );
    });
  }

  String _buildVietnameseInstruction(Map<String, dynamic> step) {
    final maneuver = step['maneuver'] as Map<String, dynamic>;
    final type = maneuver['type']?.toString() ?? '';
    final modifier = maneuver['modifier']?.toString() ?? '';
    final roadName = step['name']?.toString() ?? '';

    String action;

    if (type == 'depart') {
      action = 'Bắt đầu di chuyển';
    } else if (type == 'arrive') {
      action = 'Bạn đã đến gần thùng rác';
    } else if (modifier == 'left') {
      action = 'Rẽ trái';
    } else if (modifier == 'right') {
      action = 'Rẽ phải';
    } else if (modifier == 'slight left') {
      action = 'Chếch trái';
    } else if (modifier == 'slight right') {
      action = 'Chếch phải';
    } else if (modifier == 'straight') {
      action = 'Đi thẳng';
    } else if (modifier == 'uturn') {
      action = 'Quay đầu';
    } else {
      action = 'Tiếp tục di chuyển';
    }

    if (roadName.isNotEmpty) {
      return '$action vào $roadName';
    }

    return action;
  }

  List<Map<String, dynamic>> _extractRouteSteps(dynamic route) {
    final legs = route['legs'] as List;

    if (legs.isEmpty) return [];

    final steps = legs[0]['steps'] as List;

    return steps.map<Map<String, dynamic>>((step) {
      final maneuver = step['maneuver'] as Map<String, dynamic>;
      final location = maneuver['location'] as List;

      final lng = _toDouble(location[0]);
      final lat = _toDouble(location[1]);

      return {
        'point': LatLng(lat, lng),
        'instruction': _buildVietnameseInstruction(
          Map<String, dynamic>.from(step),
        ),
        'distance': _toDouble(step['distance']),
      };
    }).toList();
  }

  double _calculateBearing(LatLng start, LatLng end) {
    final lat1 = start.latitudeInRad;
    final lat2 = end.latitudeInRad;
    final dLng = (end.longitude - start.longitude) * 0.017453292519943295;

    final y = Math.sin(dLng) * Math.cos(lat2);
    final x = Math.cos(lat1) * Math.sin(lat2) -
        Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLng);

    final bearing = Math.atan2(y, x) * 180 / Math.pi;

    return (bearing + 360) % 360;
  }

  LatLng? _findNextRoutePoint(LatLng currentPoint) {
    if (_routePoints.length < 2) return null;

    final distance = const Distance();

    int nearestIndex = 0;
    double nearestDistance = double.infinity;

    for (int i = 0; i < _routePoints.length; i++) {
      final d = distance.as(
        LengthUnit.Meter,
        currentPoint,
        _routePoints[i],
      );

      if (d < nearestDistance) {
        nearestDistance = d;
        nearestIndex = i;
      }
    }

    final nextIndex = (nearestIndex + 5 < _routePoints.length)
        ? nearestIndex + 5
        : _routePoints.length - 1;

    return _routePoints[nextIndex];
  }

  void _updateNextInstruction(LatLng currentPoint) {
    if (_routeSteps.isEmpty) return;

    final distance = const Distance();

    if (_nextStepIndex >= _routeSteps.length) {
      setState(() {
        _nextInstructionText = 'Bạn đã đến gần thùng rác';
      });
      return;
    }

    final currentStep = _routeSteps[_nextStepIndex];
    final stepPoint = currentStep['point'] as LatLng;

    final distanceToStep = distance.as(
      LengthUnit.Meter,
      currentPoint,
      stepPoint,
    );

    if (distanceToStep < 25 && _nextStepIndex < _routeSteps.length - 1) {
      _nextStepIndex++;
    }

    final nextStep = _routeSteps[_nextStepIndex];
    final nextPoint = nextStep['point'] as LatLng;

    final nextDistance = distance.as(
      LengthUnit.Meter,
      currentPoint,
      nextPoint,
    );

    setState(() {
      _nextInstructionText =
      '${nextStep['instruction']} • còn ${nextDistance.round()} m';
    });
  }

  void _startNavigationTracking() {
    _positionSub?.cancel();

    const locationSettings = LocationSettings(
      accuracy: LocationAccuracy.high,
      distanceFilter: 3,
    );

    _positionSub = Geolocator.getPositionStream(
      locationSettings: locationSettings,
    ).listen((position) {
      if (!_isNavigating || _routePoints.isEmpty) return;

      final currentPoint = LatLng(
        position.latitude,
        position.longitude,
      );

      _updateNextInstruction(currentPoint);

      final nextPoint = _findNextRoutePoint(currentPoint);

      double bearing = _currentBearing;

      // Nếu xe/người đang di chuyển đủ nhanh thì mới dùng hướng GPS
      if (position.speed > 1.2 &&
          position.heading >= 0 &&
          position.heading <= 360) {
        bearing = position.heading;
      }
      // Nếu đứng yên hoặc heading lỗi thì dùng hướng của tuyến đường
      else if (nextPoint != null) {
        bearing = _calculateBearing(currentPoint, nextPoint);
      }

      setState(() {
        _currentLocation = currentPoint;
        _currentBearing = bearing;
        _isNavigating = true;
      });

      _mapController.moveAndRotate(
        currentPoint,
        18,
        _toMapRotation(bearing),
      );
    });
  }

  double _toMapRotation(double bearing) {
    return (360 - bearing) % 360;
  }

  Widget _buildRouteInfoBar() {
    if (_routeInfoText == null || _routePoints.isEmpty) {
      return const SizedBox.shrink();
    }

    return Positioned(
      top: 42,
      left: 20,
      right: 20,
      child: Material(
        elevation: 8,
        shadowColor: Colors.black26,
        borderRadius: BorderRadius.circular(18),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(18),
          ),
          child: Row(
            children: [
              Container(
                width: 42,
                height: 42,
                decoration: BoxDecoration(
                  color: lightGreen,
                  borderRadius: BorderRadius.circular(14),
                ),
                child: const Icon(
                  Icons.route,
                  color: primaryGreen,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Tuyến đường đến thùng rác',
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    const SizedBox(height: 3),
                    Text(
                      _nextInstructionText ?? _routeInfoText!,
                      style: const TextStyle(
                        fontSize: 13,
                        color: primaryGreen,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ],
                ),
              ),
          IconButton(
            onPressed: _clearRoute,
            icon: const Icon(Icons.close),
          ),


            ],
          ),
        ),
      ),
    );
  }

  void _fitRouteToScreen(List<LatLng> points) {
    if (points.isEmpty) return;

    final bounds = LatLngBounds.fromPoints(points);

    _mapController.fitCamera(
      CameraFit.bounds(
        bounds: bounds,
        padding: const EdgeInsets.fromLTRB(50, 130, 50, 220),
      ),
    );
  }

  Future<LatLng?> _getCurrentLocation() async {
    final serviceEnabled = await Geolocator.isLocationServiceEnabled();

    if (!serviceEnabled) {
      if (!mounted) return null;

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Bạn cần bật GPS để chỉ đường.'),
        ),
      );
      return null;
    }

    LocationPermission permission = await Geolocator.checkPermission();

    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
    }

    if (permission == LocationPermission.denied ||
        permission == LocationPermission.deniedForever) {
      if (!mounted) return null;

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('App chưa được cấp quyền vị trí.'),
        ),
      );
      return null;
    }

    final position = await Geolocator.getCurrentPosition(
      desiredAccuracy: LocationAccuracy.high,
    );

    return LatLng(position.latitude, position.longitude);
  }

  Future<void> _drawRouteToBin(Map<String, dynamic> bin) async {
    if (_isRouting) return;

    setState(() {
      _isRouting = true;
    });

    _centerBeforeRoute = _mapController.camera.center;
    _zoomBeforeRoute = _mapController.camera.zoom;

    _mapController.moveAndRotate(
      _centerBeforeRoute!,
      _zoomBeforeRoute!,
      0,
    );

    try {
      final start = await _getCurrentLocation();

      if (start == null) {
        setState(() {
          _isRouting = false;
        });
        return;
      }

      final end = LatLng(
        _toDouble(bin['latitude']),
        _toDouble(bin['longitude']),
      );

      final url =
          'https://router.project-osrm.org/route/v1/driving/'
          '${start.longitude},${start.latitude};'
          '${end.longitude},${end.latitude}'
          '?overview=full&geometries=geojson&steps=true';

      final response = await Dio().get(url);

      final data = response.data;

      if (data == null || data['routes'] == null || data['routes'].isEmpty) {
        throw Exception('Không tìm thấy tuyến đường.');
      }

      final route = data['routes'][0];
      final coordinates = route['geometry']['coordinates'] as List;
      final routeSteps = _extractRouteSteps(route);

      final distanceMeters = route['distance'] as num;
      final durationSeconds = route['duration'] as num;

      final points = coordinates.map<LatLng>((point) {
        final lng = _toDouble(point[0]);
        final lat = _toDouble(point[1]);
        return LatLng(lat, lng);
      }).toList();

      setState(() {
        _currentLocation = start;
        _routePoints = points;
        _routeSteps = routeSteps;
        _nextStepIndex = 0;
        _nextInstructionText = null;
        _routeInfoText =
        '${(distanceMeters / 1000).toStringAsFixed(1)} km • ${(durationSeconds /
            60).round()} phút';
        _selectedBinId = bin['id'];
        _isZoomedToMarker = true;
        _isNavigating = false;
      });

      await Future.delayed(const Duration(milliseconds: 250));

      _fitRouteToScreen(points);
    } catch (e) {
      if (!mounted) return;

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Lỗi chỉ đường: $e'),
        ),
      );
    } finally {
      if (mounted) {
        setState(() {
          _isRouting = false;
        });
      }
    }
  }

  Color _getStatusColor(int fillLevel) {
    if (fillLevel >= 81) return red;
    if (fillLevel >= 51) return orange;
    return primaryGreen;
  }

  Color _getStatusBgColor(int fillLevel) {
    if (fillLevel >= 81) return red.withOpacity(0.12);
    if (fillLevel >= 51) return orange.withOpacity(0.14);
    return lightGreen;
  }

  String _getStatusText(int fillLevel) {
    if (fillLevel >= 81) return 'Đầy';
    if (fillLevel >= 51) return 'Gần đầy';
    return 'Bình thường';
  }

  Map<String, dynamic>? _getSelectedBin(List<Map<String, dynamic>> bins) {
    if (_selectedBinId == null) return null;

    try {
      return bins.firstWhere((bin) => bin['id'] == _selectedBinId);
    } catch (_) {
      return null;
    }
  }

  Future<void> _animatedMove(
      LatLng destLocation,
      double destZoom, {
        int durationMs = 1000,
        Curve curve = Curves.easeInOutCubic,
      }) async {
    final latTween = Tween<double>(
      begin: _mapController.camera.center.latitude,
      end: destLocation.latitude,
    );

    final lngTween = Tween<double>(
      begin: _mapController.camera.center.longitude,
      end: destLocation.longitude,
    );

    final zoomTween = Tween<double>(
      begin: _mapController.camera.zoom,
      end: destZoom,
    );

    final controller = AnimationController(
      duration: Duration(milliseconds: durationMs),
      vsync: this,
    );

    final animation = CurvedAnimation(
      parent: controller,
      curve: curve,
    );

    void listener() {
      _mapController.move(
        LatLng(
          latTween.evaluate(animation),
          lngTween.evaluate(animation),
        ),
        zoomTween.evaluate(animation),
      );
    }

    controller.addListener(listener);

    try {
      await controller.forward();
    } finally {
      controller.removeListener(listener);
      controller.dispose();
    }
  }

  Future<void> _zoomToBin(Map<String, dynamic> bin) async {
    if (_isAnimating) return;

    _isAnimating = true;

    if (!_isZoomedToMarker) {
      _previousCenter = _mapController.camera.center;
      _previousZoom = _mapController.camera.zoom;
    }

    final target = LatLng(bin['latitude'], bin['longitude']);

    setState(() {
      _isZoomedToMarker = true;
      _selectedBinId = bin['id'];
    });

    try {
      final currentZoom = _mapController.camera.zoom;

      if (currentZoom < 12) {
        await _animatedMove(target, 12, durationMs: 600);
      }

      await _animatedMove(target, 16, durationMs: 900);
    } finally {
      _isAnimating = false;
    }
  }

  Future<void> _restorePreviousView() async {
    if (_previousCenter == null || _previousZoom == null || _isAnimating) {
      return;
    }

    _isAnimating = true;

    try {
      await _animatedMove(
        _previousCenter!,
        _previousZoom!,
        durationMs: 900,
      );

      setState(() {
        _isZoomedToMarker = false;
        _selectedBinId = null;
      });
    } finally {
      _isAnimating = false;
    }
  }

  Future<bool> handleBack() async {
    if (_showBinList) {
      setState(() => _showBinList = false);
      return true;
    }

    if (_isZoomedToMarker) {
      await _restorePreviousView();
      return true;
    }

    return false;
  }

  Widget _buildSearchBar() {
    return Positioned(
      top: 18,
      left: 20,
      right: 20,
      child: Material(
        elevation: 8,
        shadowColor: Colors.black26,
        borderRadius: BorderRadius.circular(24),
        child: TextField(
          controller: _searchController,
          onTap: () {
            setState(() {
              _showBinList = true;
            });
          },
          onChanged: (value) {
            setState(() {
              _searchText = value.trim().toLowerCase();
            });
          },
          decoration: InputDecoration(
            hintText: 'Tìm kiếm thùng rác, địa điểm...',
            hintStyle: TextStyle(
              color: Colors.grey.shade600,
              fontSize: 14,
            ),
            prefixIcon: const Icon(Icons.search),
            suffixIcon: const Icon(Icons.tune),
            filled: true,
            fillColor: Colors.white,
            contentPadding: const EdgeInsets.symmetric(vertical: 14),
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(24),
              borderSide: BorderSide.none,
            ),
          ),
        ),
      ),
    );
  }


  Widget _buildRightMapButtons() {
    return Positioned(
      right: 18,
      top: MediaQuery.of(context).padding.top + 88,
      child: _roundMapButton(
        icon: Icons.my_location,
        onTap: () {
          _mapController.move(
            const LatLng(16.0544, 108.2022),
            13,
          );
        },
      ),
    );
  }

  Widget _roundMapButton({
    required IconData icon,
    required VoidCallback onTap,
    bool isBig = false,
  }) {
    return Material(
      elevation: 8,
      shadowColor: Colors.black26,
      shape: const CircleBorder(),
      child: InkWell(
        customBorder: const CircleBorder(),
        onTap: onTap,
        child: Container(
          width: isBig ? 56 : 48,
          height: isBig ? 56 : 48,
          decoration: BoxDecoration(
            color: isBig ? primaryGreen : Colors.white,
            shape: BoxShape.circle,
          ),
          child: Icon(
            icon,
            color: isBig ? Colors.white : primaryGreen,
            size: isBig ? 30 : 25,
          ),
        ),
      ),
    );
  }

  Widget _buildMarker(Map<String, dynamic> bin, bool isSelected) {
    final fillLevel = bin['fillLevel'] as int;
    final color = _getStatusColor(fillLevel);

    return GestureDetector(
      onTap: () => _zoomToBin(bin),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (isSelected)
            Container(
              margin: const EdgeInsets.only(bottom: 3),
              padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 5),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(9),
                boxShadow: const [
                  BoxShadow(
                    color: Colors.black26,
                    blurRadius: 5,
                  ),
                ],
              ),
              child: Text(
                bin['name'],
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
          Stack(
            alignment: Alignment.center,
            children: [
              Icon(
                Icons.location_on,
                size: isSelected ? 52 : 46,
                color: color,
              ),
              Positioned(
                top: isSelected ? 11 : 10,
                child: Icon(
                  Icons.delete_outline,
                  color: Colors.white,
                  size: isSelected ? 20 : 17,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildBottomInfoCard(Map<String, dynamic> bin) {
    final fillLevel = bin['fillLevel'] as int;
    final statusColor = _getStatusColor(fillLevel);
    final statusText = _getStatusText(fillLevel);

    return Positioned(
      left: 0,
      right: 0,
      bottom: 0,
      child: Container(
        padding: const EdgeInsets.fromLTRB(22, 14, 22, 18),
        decoration: const BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.vertical(
            top: Radius.circular(26),
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.black26,
              blurRadius: 18,
              offset: Offset(0, -4),
            ),
          ],
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 48,
              height: 5,
              margin: const EdgeInsets.only(bottom: 16),
              decoration: BoxDecoration(
                color: Colors.grey.shade300,
                borderRadius: BorderRadius.circular(20),
              ),
            ),
            Row(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                Container(
                  width: 68,
                  height: 68,
                  decoration: BoxDecoration(
                    color: lightGreen,
                    borderRadius: BorderRadius.circular(22),
                  ),
                  child: const Icon(
                    Icons.delete,
                    color: primaryGreen,
                    size: 38,
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        bin['name'],
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.w800,
                        ),
                      ),
                      const SizedBox(height: 7),
                      Row(
                        children: [
                          const Icon(
                            Icons.location_on_outlined,
                            size: 16,
                            color: Colors.black54,
                          ),
                          const SizedBox(width: 5),
                          Expanded(
                            child: Text(
                              bin['location'],
                              maxLines: 2,
                              overflow: TextOverflow.ellipsis,
                              style: const TextStyle(
                                fontSize: 13,
                                color: Colors.black87,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 10),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 10,
                        vertical: 6,
                      ),
                      decoration: BoxDecoration(
                        color: _getStatusBgColor(fillLevel),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        statusText,
                        style: TextStyle(
                          color: statusColor,
                          fontSize: 12,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ),
                    const SizedBox(height: 10),
                    Text(
                      '$fillLevel%',
                      style: const TextStyle(
                        fontSize: 19,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    const Text(
                      'Mức rác',
                      style: TextStyle(
                        fontSize: 12,
                        color: Colors.black54,
                      ),
                    ),
                    const SizedBox(height: 5),
                    SizedBox(
                      width: 55,
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(10),
                        child: LinearProgressIndicator(
                          value: fillLevel / 100,
                          minHeight: 4,
                          backgroundColor: Colors.grey.shade200,
                          valueColor: AlwaysStoppedAnimation(statusColor),
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
            const SizedBox(height: 18),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () {},
                    icon: const Icon(Icons.info, size: 19),
                    label: const Text('Xem chi tiết'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: primaryGreen,
                      foregroundColor: Colors.white,
                      elevation: 0,
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(14),
                      ),
                      textStyle: const TextStyle(
                        fontWeight: FontWeight.w700,
                        fontSize: 15,
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: _isRouting
                        ? null
                        : () {
                      _drawRouteToBin(bin);
                    },
                    icon: _isRouting
                        ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                        : const Icon(Icons.navigation, size: 19),
                    label: Text(_isRouting ? 'Đang tìm...' : 'Dẫn đường'),
                    style: OutlinedButton.styleFrom(
                      foregroundColor: primaryGreen,
                      side: const BorderSide(color: primaryGreen, width: 1.2),
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(14),
                      ),
                      textStyle: const TextStyle(
                        fontWeight: FontWeight.w700,
                        fontSize: 15,
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }



  Widget _buildBinListOverlay(List<Map<String, dynamic>> bins) {
    final filteredBins = bins.where((bin) {
      final name = bin['name'].toString().toLowerCase();
      final location = bin['location'].toString().toLowerCase();

      return name.contains(_searchText) || location.contains(_searchText);
    }).toList();

    return Positioned.fill(
      child: Stack(
        children: [
          GestureDetector(
            onTap: () {
              setState(() {
                _showBinList = false;
              });
            },
            child: Container(
              color: Colors.black.withOpacity(0.55),
            ),
          ),
          Align(
            alignment: Alignment.bottomCenter,
            child: Container(
              height: MediaQuery.of(context).size.height * 0.64,
              padding: const EdgeInsets.fromLTRB(20, 10, 20, 20),
              decoration: const BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.vertical(
                  top: Radius.circular(28),
                ),
              ),
              child: Column(
                children: [
                  Container(
                    width: 46,
                    height: 5,
                    margin: const EdgeInsets.only(bottom: 16),
                    decoration: BoxDecoration(
                      color: Colors.grey.shade300,
                      borderRadius: BorderRadius.circular(20),
                    ),
                  ),
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              'Danh sách thùng rác',
                              style: TextStyle(
                                fontSize: 22,
                                fontWeight: FontWeight.w800,
                              ),
                            ),
                            const SizedBox(height: 3),
                            Text(
                              '${filteredBins.length} thùng rác',
                              style: TextStyle(
                                color: Colors.grey.shade600,
                                fontSize: 14,
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                          ],
                        ),
                      ),
                      InkWell(
                        customBorder: const CircleBorder(),
                        onTap: () {
                          setState(() {
                            _showBinList = false;
                          });
                        },
                        child: Container(
                          width: 44,
                          height: 44,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            border: Border.all(color: Colors.grey.shade300),
                          ),
                          child: const Icon(Icons.close),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  TextField(
                    controller: _searchController,
                    onChanged: (value) {
                      setState(() {
                        _searchText = value.trim().toLowerCase();
                      });
                    },
                    decoration: InputDecoration(
                      hintText: 'Tìm kiếm...',
                      prefixIcon: const Icon(Icons.search),
                      filled: true,
                      fillColor: Colors.white,
                      contentPadding: const EdgeInsets.symmetric(vertical: 13),
                      enabledBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(13),
                        borderSide: BorderSide(color: Colors.grey.shade300),
                      ),
                      focusedBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(13),
                        borderSide: const BorderSide(color: primaryGreen),
                      ),
                    ),
                  ),
                  const SizedBox(height: 14),
                  Expanded(
                    child: ListView.separated(
                      padding: EdgeInsets.zero,
                      itemCount: filteredBins.length,
                      separatorBuilder: (_, __) => Divider(
                        height: 1,
                        color: Colors.grey.shade200,
                      ),
                      itemBuilder: (context, index) {
                        final bin = filteredBins[index];
                        final fillLevel = bin['fillLevel'] as int;
                        final statusColor = _getStatusColor(fillLevel);
                        final statusText = _getStatusText(fillLevel);
                        final isSelected = _selectedBinId == bin['id'];

                        return InkWell(
                          borderRadius: BorderRadius.circular(14),
                          onTap: () async {
                            setState(() {
                              _selectedBinId = bin['id'];
                              _showBinList = false;
                            });

                            await _zoomToBin(bin);
                          },
                          child: Container(
                            margin: const EdgeInsets.symmetric(vertical: 6),
                            padding: const EdgeInsets.symmetric(
                              horizontal: 12,
                              vertical: 12,
                            ),
                            decoration: BoxDecoration(
                              color: isSelected ? lightGreen : Colors.white,
                              borderRadius: BorderRadius.circular(15),
                            ),
                            child: Row(
                              children: [
                                Container(
                                  width: 48,
                                  height: 48,
                                  decoration: BoxDecoration(
                                    color: _getStatusBgColor(fillLevel),
                                    borderRadius: BorderRadius.circular(15),
                                  ),
                                  child: Icon(
                                    Icons.delete,
                                    color: statusColor,
                                  ),
                                ),
                                const SizedBox(width: 13),
                                Expanded(
                                  child: Column(
                                    crossAxisAlignment:
                                    CrossAxisAlignment.start,
                                    children: [
                                      Row(
                                        children: [
                                          Flexible(
                                            child: Text(
                                              bin['name'],
                                              maxLines: 1,
                                              overflow: TextOverflow.ellipsis,
                                              style: const TextStyle(
                                                fontSize: 16,
                                                fontWeight: FontWeight.w800,
                                              ),
                                            ),
                                          ),
                                          if (isSelected) ...[
                                            const SizedBox(width: 10),
                                            const Icon(
                                              Icons.circle,
                                              color: primaryGreen,
                                              size: 7,
                                            ),
                                            const SizedBox(width: 4),
                                            const Text(
                                              'Đang chọn',
                                              style: TextStyle(
                                                color: primaryGreen,
                                                fontSize: 12,
                                                fontWeight: FontWeight.w700,
                                              ),
                                            ),
                                          ],
                                        ],
                                      ),
                                      const SizedBox(height: 5),
                                      Text(
                                        bin['location'],
                                        maxLines: 1,
                                        overflow: TextOverflow.ellipsis,
                                        style: TextStyle(
                                          color: Colors.grey.shade700,
                                          fontSize: 13,
                                          fontWeight: FontWeight.w500,
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                                const SizedBox(width: 10),
                                Column(
                                  crossAxisAlignment: CrossAxisAlignment.end,
                                  children: [
                                    Text(
                                      '$fillLevel%',
                                      style: TextStyle(
                                        color: statusColor,
                                        fontWeight: FontWeight.w800,
                                        fontSize: 16,
                                      ),
                                    ),
                                    const SizedBox(height: 4),
                                    Text(
                                      statusText,
                                      style: TextStyle(
                                        color: statusColor,
                                        fontSize: 12,
                                        fontWeight: FontWeight.w700,
                                      ),
                                    ),
                                  ],
                                ),
                              ],
                            ),
                          ),
                        );
                      },
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  LatLng _getInitialCenter(List<Map<String, dynamic>> bins) {
    if (widget.initialBinId != null) {
      try {
        final targetBin =
        bins.firstWhere((bin) => bin['id'] == widget.initialBinId);
        return LatLng(targetBin['latitude'], targetBin['longitude']);
      } catch (_) {}
    }

    return const LatLng(16.0544, 108.2022);
  }

  double _getInitialZoom() {
    if (widget.initialBinId != null) return 16;
    return 12;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: FutureBuilder<List<Map<String, dynamic>>>(
        future: _binsFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(
              child: CircularProgressIndicator(color: primaryGreen),
            );
          }

          if (snapshot.hasError) {
            return Center(
              child: Text('Lỗi tải dữ liệu: ${snapshot.error}'),
            );
          }

          if (!snapshot.hasData || snapshot.data!.isEmpty) {
            return const Center(
              child: Text('Chưa có thùng rác nào có tọa độ.'),
            );
          }

          final bins = snapshot.data!;
          final selectedBin = _getSelectedBin(bins);

          return Stack(
            children: [
              FlutterMap(
                mapController: _mapController,
                options: MapOptions(
                  initialCenter: _getInitialCenter(bins),
                  initialZoom: _getInitialZoom(),
                ),
                children: [
                  TileLayer(
                    urlTemplate:
                    'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                    userAgentPackageName: 'com.example.pbl5Flutter',
                    keepBuffer: 4,
                    panBuffer: 1,
                    maxNativeZoom: 19,
                    tileDisplay: TileDisplay.fadeIn(),
                  ),
                  if (_routePoints.isNotEmpty)
                    PolylineLayer(
                      polylines: [
                        Polyline(
                          points: _routePoints,
                          strokeWidth: 10,
                          color: Colors.white.withOpacity(0.92),
                        ),
                        Polyline(
                          points: _routePoints,
                          strokeWidth: 6,
                          color: primaryGreen,
                        ),
                      ],
                    ),
                  MarkerLayer(
                    markers: bins.map((bin) {
                      final isSelected = _selectedBinId == bin['id'];

                      return Marker(
                        point: LatLng(bin['latitude'], bin['longitude']),
                        width: 120,
                        height: isSelected ? 88 : 56,
                        child: _buildMarker(bin, isSelected),
                      );
                    }).toList(),
                  ),
                  if (_currentLocation != null)
                    MarkerLayer(
                      markers: [
                        Marker(
                          point: _currentLocation!,
                          width: 56,
                          height: 56,
                          child: Container(
                            decoration: BoxDecoration(
                              color: lightGreen,
                              shape: BoxShape.circle,
                            ),
                            child: Center(
                              child: Transform.rotate(
                                angle: 0,
                                child: const Icon(
                                  Icons.navigation,
                                  color: primaryGreen,
                                  size: 32,
                                ),
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                ],
              ),

              if (!_isNavigating)
                SafeArea(
                  child: Stack(
                    children: [
                      _buildSearchBar(),
                    ],
                  ),
                ),

              if (!_isNavigating)
                _buildRightMapButtons(),

              if (!_showBinList && _routePoints.isNotEmpty && _isNavigating)
                _buildRouteInfoBar(),

              if (selectedBin != null && !_showBinList && _routePoints.isEmpty)
                _buildBottomInfoCard(selectedBin),

              if (!_showBinList && _routePoints.isNotEmpty && !_isNavigating)
                _buildRoutePreviewPanel(selectedBin),

              if (_showBinList)
                _buildBinListOverlay(bins),
            ],
          );
        },
      ),
    );
  }
}
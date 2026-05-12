import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';

import '../utils/top_toast.dart';

class LocationPickerScreen extends StatefulWidget {
  const LocationPickerScreen({
    super.key,
    this.initialLocation,
  });

  final LatLng? initialLocation;

  @override
  State<LocationPickerScreen> createState() => _LocationPickerScreenState();
}

class _LocationPickerScreenState extends State<LocationPickerScreen> {
  final MapController _mapController = MapController();
  final TextEditingController _searchCtrl = TextEditingController();
  final Dio _dio = Dio();

  LatLng? _selectedPoint;
  bool _isSearching = false;

  List<GeoSearchResult> _searchResults = [];

  static const darkGreen = Color(0xFF0B5D1E);
  static const bgColor = Color(0xFFFBFDF7);

  // Default: Đà Nẵng
  static const LatLng _defaultPoint = LatLng(16.047079, 108.206230);

  @override
  void initState() {
    super.initState();
    _selectedPoint = widget.initialLocation;
  }

  @override
  void dispose() {
    _searchCtrl.dispose();
    super.dispose();
  }

  Future<void> _searchLocation() async {
    final keyword = _searchCtrl.text.trim();

    if (keyword.isEmpty) {
      TopToast.show(
        context,
        'Please enter location to search.',
        type: ToastType.error,
      );
      return;
    }

    FocusManager.instance.primaryFocus?.unfocus();

    setState(() {
      _isSearching = true;
      _searchResults = [];
    });

    try {
      final lowerKeyword = keyword.toLowerCase();

      final searchText = lowerKeyword.contains('vietnam') ||
          lowerKeyword.contains('việt nam') ||
          lowerKeyword.contains('đà nẵng') ||
          lowerKeyword.contains('da nang')
          ? keyword
          : '$keyword, Đà Nẵng, Việt Nam';

      final response = await _dio.get(
        'https://nominatim.openstreetmap.org/search',
        queryParameters: {
          'q': searchText,
          'format': 'jsonv2',
          'addressdetails': 1,
          'limit': 8,
          'countrycodes': 'vn',
        },
        options: Options(
          headers: {
            'User-Agent': 'SmartBinStudentApp/1.0',
          },
        ),
      );

      final data = response.data as List;

      if (!mounted) return;

      if (data.isEmpty) {
        TopToast.show(
          context,
          'Location not found.',
          type: ToastType.error,
        );
        return;
      }

      final uniqueResults = <String, GeoSearchResult>{};

      for (final item in data) {
        final lat = double.parse(item['lat'].toString());
        final lon = double.parse(item['lon'].toString());

        final displayName =
            item['display_name']?.toString() ?? 'Selected location';

        final key = displayName.toLowerCase().trim();

        if (!uniqueResults.containsKey(key)) {
          uniqueResults[key] = GeoSearchResult(
            point: LatLng(lat, lon),
            label: displayName,
          );
        }
      }

      setState(() {
        _searchResults = uniqueResults.values.take(6).toList();
      });
    } catch (e) {
      if (!mounted) return;

      TopToast.show(
        context,
        'Location not found. Try another keyword.',
        type: ToastType.error,
      );
    } finally {
      if (mounted) {
        setState(() => _isSearching = false);
      }
    }
  }

  void _selectSearchResult(GeoSearchResult item) {
    FocusManager.instance.primaryFocus?.unfocus();

    setState(() {
      _selectedPoint = item.point;
      _searchCtrl.text = item.label;
      _searchResults = [];
    });

    _mapController.move(item.point, 16);
  }

  Future<String> _getAddressFromPoint(LatLng point) async {
    try {
      final response = await _dio.get(
        'https://nominatim.openstreetmap.org/reverse',
        queryParameters: {
          'lat': point.latitude,
          'lon': point.longitude,
          'format': 'jsonv2',
          'addressdetails': 1,
          'zoom': 18,
          'accept-language': 'vi',
        },
        options: Options(
          headers: {
            'User-Agent': 'SmartBinStudentApp/1.0',
          },
        ),
      );

      final displayName = response.data['display_name']?.toString().trim();

      if (displayName == null || displayName.isEmpty) {
        return 'Selected location';
      }

      return displayName;
    } catch (e) {
      return 'Selected location';
    }
  }

  Future<void> _selectPointFromMap(LatLng point) async {
    FocusManager.instance.primaryFocus?.unfocus();

    setState(() {
      _selectedPoint = point;
      _searchResults = [];
      _searchCtrl.text = 'Finding address...';
    });

    final address = await _getAddressFromPoint(point);

    if (!mounted) return;

    setState(() {
      _searchCtrl.text = address;
    });
  }

  void _confirmLocation() {
    if (_selectedPoint == null) {
      TopToast.show(
        context,
        'Please select a location first.',
        type: ToastType.error,
      );
      return;
    }

    Navigator.pop(context, _selectedPoint);
  }

  @override
  Widget build(BuildContext context) {
    final center = _selectedPoint ?? widget.initialLocation ?? _defaultPoint;

    return Scaffold(
      backgroundColor: bgColor,
      body: SafeArea(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 10, 16, 10),
              child: Row(
                children: [
                  IconButton(
                    onPressed: () {
                      FocusManager.instance.primaryFocus?.unfocus();
                      Navigator.pop(context);
                    },
                    icon: const Icon(Icons.arrow_back, size: 28),
                  ),
                  const SizedBox(width: 4),
                  const Expanded(
                    child: Text(
                      'Pick Location',
                      style: TextStyle(
                        fontSize: 28,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                ],
              ),
            ),

            Padding(
              padding: const EdgeInsets.fromLTRB(18, 0, 18, 12),
              child: Column(
                children: [
                  Container(
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(18),
                      border: Border.all(
                        color: const Color(0xFFE5EEDC),
                      ),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withOpacity(0.06),
                          blurRadius: 16,
                          offset: const Offset(0, 8),
                        ),
                      ],
                    ),
                    child: TextField(
                      controller: _searchCtrl,
                      textInputAction: TextInputAction.search,
                      onSubmitted: (_) => _searchLocation(),
                      onTapOutside: (_) {
                        FocusManager.instance.primaryFocus?.unfocus();
                      },
                      decoration: InputDecoration(
                        prefixIcon: const Icon(
                          Icons.search,
                          color: darkGreen,
                        ),
                        hintText: 'Search address or place',
                        border: InputBorder.none,
                        contentPadding: const EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 15,
                        ),
                        suffixIcon: _isSearching
                            ? const Padding(
                          padding: EdgeInsets.all(14),
                          child: SizedBox(
                            width: 18,
                            height: 18,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                            ),
                          ),
                        )
                            : IconButton(
                          onPressed: _searchLocation,
                          icon: const Icon(
                            Icons.search,
                            color: darkGreen,
                          ),
                        ),
                      ),
                    ),
                  ),

                  if (_searchResults.isNotEmpty)
                    Container(
                      margin: const EdgeInsets.only(top: 10),
                      constraints: const BoxConstraints(
                        maxHeight: 310,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(18),
                        border: Border.all(
                          color: const Color(0xFFE5EEDC),
                        ),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withOpacity(0.08),
                            blurRadius: 16,
                            offset: const Offset(0, 8),
                          ),
                        ],
                      ),
                      child: ListView.separated(
                        shrinkWrap: true,
                        padding: EdgeInsets.zero,
                        itemCount: _searchResults.length,
                        separatorBuilder: (_, __) => Divider(
                          height: 1,
                          color: Colors.black.withOpacity(0.08),
                        ),
                        itemBuilder: (context, index) {
                          final item = _searchResults[index];

                          return ListTile(
                            contentPadding: const EdgeInsets.symmetric(
                              horizontal: 14,
                              vertical: 6,
                            ),
                            leading: Container(
                              width: 42,
                              height: 42,
                              decoration: BoxDecoration(
                                color: const Color(0xFFEAF6C8),
                                borderRadius: BorderRadius.circular(14),
                              ),
                              child: const Icon(
                                Icons.location_on_outlined,
                                color: Color(0xFF2F6B3D),
                              ),
                            ),
                            title: Text(
                              item.label,
                              maxLines: 3,
                              overflow: TextOverflow.ellipsis,
                              style: const TextStyle(
                                fontSize: 15,
                                fontWeight: FontWeight.w800,
                                color: Colors.black87,
                                height: 1.35,
                              ),
                            ),
                            trailing: const Icon(
                              Icons.north_west,
                              color: Colors.black54,
                            ),
                            onTap: () => _selectSearchResult(item),
                          );
                        },
                      ),
                    ),
                ],
              ),
            ),

            Expanded(
              child: Padding(
                padding: const EdgeInsets.fromLTRB(18, 0, 18, 14),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(22),
                  child: FlutterMap(
                    mapController: _mapController,
                    options: MapOptions(
                      initialCenter: center,
                      initialZoom: widget.initialLocation == null ? 13 : 16,
                      onTap: (tapPosition, point) {
                        _selectPointFromMap(point);
                      },
                    ),
                    children: [
                      TileLayer(
                        urlTemplate:
                        'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                        userAgentPackageName: 'com.example.pbl5Flutter',
                      ),

                      if (_selectedPoint != null)
                        CircleLayer(
                          circles: [
                            CircleMarker(
                              point: _selectedPoint!,
                              radius: 48,
                              color: const Color(0xFF4CAF50).withOpacity(0.18),
                              borderStrokeWidth: 0,
                            ),
                          ],
                        ),

                      if (_selectedPoint != null)
                        MarkerLayer(
                          markers: [
                            Marker(
                              point: _selectedPoint!,
                              width: 56,
                              height: 56,
                              child: const Icon(
                                Icons.location_pin,
                                color: Color(0xFF2F9E44),
                                size: 52,
                              ),
                            ),
                          ],
                        ),
                    ],
                  ),
                ),
              ),
            ),

            Padding(
              padding: const EdgeInsets.fromLTRB(18, 0, 18, 18),
              child: SizedBox(
                width: double.infinity,
                height: 56,
                child: ElevatedButton.icon(
                  onPressed: _confirmLocation,
                  icon: const Icon(Icons.check),
                  label: const Text(
                    'Confirm Location',
                    style: TextStyle(
                      fontSize: 17,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: darkGreen,
                    foregroundColor: Colors.white,
                    elevation: 0,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(18),
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class GeoSearchResult {
  final LatLng point;
  final String label;

  const GeoSearchResult({
    required this.point,
    required this.label,
  });
}
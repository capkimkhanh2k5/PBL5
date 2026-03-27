import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';
import 'package:geolocator/geolocator.dart';

class LocationPickerScreen extends StatefulWidget {
  const LocationPickerScreen({super.key, this.initialLocation});

  final LatLng? initialLocation;

  @override
  State<LocationPickerScreen> createState() => _LocationPickerScreenState();
}

class _LocationPickerScreenState extends State<LocationPickerScreen> {
  static const _fallback = LatLng(16.0544, 108.2022);

  final MapController _mapController = MapController();
  LatLng? _selected;
  bool _loadingCurrent = false;

  LatLng get _initial => widget.initialLocation ?? _fallback;

  @override
  void initState() {
    super.initState();
    _selected = widget.initialLocation;
  }

  Future<void> _pickCurrentLocation() async {
    setState(() => _loadingCurrent = true);
    try {
      final serviceEnabled = await Geolocator.isLocationServiceEnabled();
      if (!serviceEnabled) {
        throw Exception('Location service is disabled.');
      }

      var permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
      }
      if (permission == LocationPermission.denied ||
          permission == LocationPermission.deniedForever) {
        throw Exception('Location permission is denied.');
      }

      final position = await Geolocator.getCurrentPosition(
        locationSettings: const LocationSettings(
          accuracy: LocationAccuracy.high,
        ),
      );

      final point = LatLng(position.latitude, position.longitude);
      _mapController.move(point, 16);
      setState(() => _selected = point);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(e.toString())),
      );
    } finally {
      if (mounted) {
        setState(() => _loadingCurrent = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Pick Location'),
        actions: [
          IconButton(
            onPressed: _loadingCurrent ? null : _pickCurrentLocation,
            icon: _loadingCurrent
                ? const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.my_location),
            tooltip: 'Use current location',
          ),
        ],
      ),
      body: FlutterMap(
        mapController: _mapController,
        options: MapOptions(
          initialCenter: _initial,
          initialZoom: 14,
          onTap: (_, point) => setState(() => _selected = point),
        ),
        children: [
          TileLayer(
            urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
            userAgentPackageName: 'com.example.pbl5Flutter',
          ),
          if (_selected != null)
            MarkerLayer(
              markers: [
                Marker(
                  width: 52,
                  height: 52,
                  point: _selected!,
                  child: const Icon(
                    Icons.location_pin,
                    color: Colors.red,
                    size: 44,
                  ),
                ),
              ],
            ),
        ],
      ),
      bottomNavigationBar: SafeArea(
        minimum: const EdgeInsets.fromLTRB(16, 8, 16, 16),
        child: ElevatedButton.icon(
          onPressed: _selected == null
              ? null
              : () => Navigator.pop(context, _selected),
          icon: const Icon(Icons.check_circle_outline),
          label: Text(
            _selected == null ? 'Tap on map to select' : 'Confirm this location',
          ),
          style: ElevatedButton.styleFrom(
            minimumSize: const Size.fromHeight(48),
          ),
        ),
      ),
    );
  }
}

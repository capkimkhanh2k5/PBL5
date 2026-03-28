import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import '../utils/top_toast.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';
import 'package:geolocator/geolocator.dart';

import '../services/api_service.dart';
import '../services/auth_service.dart';
import 'location_picker_screen.dart';

class BinConnectionFormScreen extends StatefulWidget {
  const BinConnectionFormScreen({super.key, required this.binId});

  final String binId;

  @override
  State<BinConnectionFormScreen> createState() => _BinConnectionFormScreenState();
}

class _BinConnectionFormScreenState extends State<BinConnectionFormScreen> {
  final _formKey = GlobalKey<FormState>();
  final _nameCtrl = TextEditingController();
  final _descCtrl = TextEditingController();
  final _authService = AuthService();

  bool _saving = false;
  double? _latitude;
  double? _longitude;

  @override
  void initState() {
    super.initState();
    _nameCtrl.text = 'Bin ${widget.binId}';
  }

  @override
  void dispose() {
    _nameCtrl.dispose();
    _descCtrl.dispose();
    super.dispose();
  }

  Future<void> _useCurrentLocation() async {
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

      final pos = await Geolocator.getCurrentPosition(
        locationSettings: const LocationSettings(
          accuracy: LocationAccuracy.high,
        ),
      );

      if (!mounted) return;
      setState(() {
        _latitude = pos.latitude;
        _longitude = pos.longitude;
      });
    } catch (e) {
      if (!mounted) return;
      TopToast.show(context, e.toString();
    }
  }

  Future<void> _pickOnMap() async {
    final initial = (_latitude != null && _longitude != null)
        ? LatLng(_latitude!, _longitude!)
        : null;

    final picked = await Navigator.push<LatLng>(
      context,
      MaterialPageRoute(
        builder: (_) => LocationPickerScreen(initialLocation: initial),
      ),
    );

    if (picked == null || !mounted) return;

    setState(() {
      _latitude = picked.latitude;
      _longitude = picked.longitude;
    });
  }

  Future<void> _submit() async {
    final valid = _formKey.currentState?.validate() ?? false;
    if (!valid) return;

    if (_latitude == null || _longitude == null) {
      TopToast.show(context, 'Please select location first.');
      return;
    }

    final confirmed = await _showConfirmDialog();
    if (confirmed != true) return;

    setState(() => _saving = true);

    try {
      final api = ApiService(authService: _authService);
      await api.connectBinFromQr(
        binId: widget.binId,
        name: _nameCtrl.text,
        locationDescription: _descCtrl.text,
        latitude: _latitude!,
        longitude: _longitude!,
      );

      if (!mounted) return;
      TopToast.show(context, 'Bin connected successfully.');
      Navigator.pop(context, true);
    } catch (e) {
      if (!mounted) return;

      var message = 'Failed to connect bin.';
      if (e is DioException) {
        message =
            e.response?.data?.toString() ?? e.message ?? 'Failed to connect bin.';
      } else {
        message = e.toString();
      }

      TopToast.show(context, message);
    } finally {
      if (mounted) {
        setState(() => _saving = false);
      }
    }
  }

  Future<bool?> _showConfirmDialog() {
    final point = LatLng(_latitude!, _longitude!);
    final name = _nameCtrl.text.trim();
    final desc = _descCtrl.text.trim();

    return showDialog<bool>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Confirm Bin Connection'),
          content: SizedBox(
            width: 340,
            child: SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text('Bin ID (from QR): ${widget.binId}'),
                  const SizedBox(height: 6),
                  Text('Name: $name'),
                  if (desc.isNotEmpty) ...[
                    const SizedBox(height: 6),
                    Text('Description: $desc'),
                  ],
                  const SizedBox(height: 6),
                  Text(
                    'Lat/Lng: ${_latitude!.toStringAsFixed(6)}, ${_longitude!.toStringAsFixed(6)}',
                  ),
                  const SizedBox(height: 10),
                  SizedBox(
                    height: 170,
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(10),
                      child: FlutterMap(
                        options: MapOptions(
                          initialCenter: point,
                          initialZoom: 15,
                          interactionOptions: const InteractionOptions(
                            flags: InteractiveFlag.none,
                          ),
                        ),
                        children: [
                          TileLayer(
                            urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                            userAgentPackageName: 'com.example.pbl5Flutter',
                          ),
                          MarkerLayer(
                            markers: [
                              Marker(
                                point: point,
                                width: 48,
                                height: 48,
                                child: const Icon(
                                  Icons.location_pin,
                                  color: Colors.red,
                                  size: 40,
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Back'),
            ),
            ElevatedButton(
              onPressed: () => Navigator.pop(context, true),
              child: const Text('Confirm'),
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    final locationText = (_latitude != null && _longitude != null)
        ? 'Lat: ${_latitude!.toStringAsFixed(6)}, Lng: ${_longitude!.toStringAsFixed(6)}'
        : 'No location selected yet';

    return Scaffold(
      appBar: AppBar(
        title: const Text('Connect New Bin'),
      ),
      body: Form(
        key: _formKey,
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Card(
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Row(
                  children: [
                    const Icon(Icons.qr_code_2, color: Colors.green),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            'Bin ID from QR',
                            style: TextStyle(fontWeight: FontWeight.w700),
                          ),
                          const SizedBox(height: 2),
                          Text(widget.binId),
                          const SizedBox(height: 2),
                          const Text(
                            'ID is auto-read from QR, users do not need to type it.',
                            style: TextStyle(fontSize: 12, color: Colors.black54),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 12),
            TextFormField(
              controller: _nameCtrl,
              decoration: const InputDecoration(
                labelText: 'Bin Name',
                border: OutlineInputBorder(),
              ),
              validator: (v) {
                if (v == null || v.trim().isEmpty) {
                  return 'Please enter bin name';
                }
                return null;
              },
            ),
            const SizedBox(height: 12),
            TextFormField(
              controller: _descCtrl,
              maxLines: 2,
              decoration: const InputDecoration(
                labelText: 'Location Description',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 16),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Bin Position',
                      style: TextStyle(fontWeight: FontWeight.w700),
                    ),
                    const SizedBox(height: 8),
                    Text(locationText),
                    const SizedBox(height: 12),
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: [
                        OutlinedButton.icon(
                          onPressed: _saving ? null : _useCurrentLocation,
                          icon: const Icon(Icons.my_location),
                          label: const Text('Use current location'),
                        ),
                        OutlinedButton.icon(
                          onPressed: _saving ? null : _pickOnMap,
                          icon: const Icon(Icons.map_outlined),
                          label: const Text('Pick on map'),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),
            ElevatedButton.icon(
              onPressed: _saving ? null : _submit,
              icon: _saving
                  ? const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.link),
              label: Text(_saving ? 'Connecting...' : 'Complete Connection'),
              style: ElevatedButton.styleFrom(
                minimumSize: const Size.fromHeight(48),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

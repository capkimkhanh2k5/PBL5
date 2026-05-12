import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:geolocator/geolocator.dart';
import 'package:latlong2/latlong.dart';

import '../services/api_service.dart';
import '../services/auth_service.dart';
import '../utils/top_toast.dart';
import 'location_picker_screen.dart';
import 'package:geocoding/geocoding.dart';

class BinConnectionFormScreen extends StatefulWidget {
  const BinConnectionFormScreen({
    super.key,
    required this.binId,
  });

  final String binId;

  @override
  State<BinConnectionFormScreen> createState() =>
      _BinConnectionFormScreenState();
}

class _BinConnectionFormScreenState extends State<BinConnectionFormScreen> {
  final _formKey = GlobalKey<FormState>();
  final _nameCtrl = TextEditingController();
  final _descCtrl = TextEditingController();

  final _authService = AuthService();

  bool _saving = false;
  double? _latitude;
  double? _longitude;
  String? _addressText;
  bool _isResolvingAddress = false;

  static const darkGreen = Color(0xFF0B5D1E);
  static const softGreen = Color(0xFFEAF6C8);
  static const borderGreen = Color(0xFF4B8B3B);
  static const bgColor = Color(0xFFFBFDF7);

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

  String _formatPlacemark(Placemark place) {
    final road = [
      place.subThoroughfare,
      place.thoroughfare,
    ].where((e) => e != null && e.trim().isNotEmpty).join(' ');

    final parts = <String>[
      if (road.trim().isNotEmpty) road,
      if (road.trim().isEmpty && (place.street ?? '').trim().isNotEmpty)
        place.street!,
      if ((place.subLocality ?? '').trim().isNotEmpty) place.subLocality!,
      if ((place.locality ?? '').trim().isNotEmpty) place.locality!,
      if ((place.administrativeArea ?? '').trim().isNotEmpty)
        place.administrativeArea!,
      if ((place.country ?? '').trim().isNotEmpty) place.country!,
    ];

    final uniqueParts = <String>[];

    for (final part in parts) {
      final clean = part.trim();
      if (clean.isNotEmpty && !uniqueParts.contains(clean)) {
        uniqueParts.add(clean);
      }
    }

    if (uniqueParts.isEmpty) {
      return 'Selected location';
    }

    return uniqueParts.join(', ');
  }

  Future<void> _loadAddressFromLatLng(double lat, double lng) async {
    setState(() {
      _isResolvingAddress = true;
      _addressText = null;
    });

    try {
      final places = await placemarkFromCoordinates(lat, lng);

      if (!mounted) return;

      if (places.isNotEmpty) {
        setState(() {
          _addressText = _formatPlacemark(places.first);
        });
      } else {
        setState(() {
          _addressText = 'Selected location';
        });
      }
    } catch (e) {
      if (!mounted) return;

      setState(() {
        _addressText = 'Selected location';
      });
    } finally {
      if (mounted) {
        setState(() {
          _isResolvingAddress = false;
        });
      }
    }
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

      await _loadAddressFromLatLng(pos.latitude, pos.longitude);
    } catch (e) {
      if (!mounted) return;
      TopToast.show(
        context,
        e.toString(),
        type: ToastType.error,
      );
    }
  }

  Future<void> _pickOnMap() async {
    final initial = (_latitude != null && _longitude != null)
        ? LatLng(_latitude!, _longitude!)
        : null;

    final picked = await Navigator.push<LatLng>(
      context,
      MaterialPageRoute(
        builder: (_) => LocationPickerScreen(
          initialLocation: initial,
        ),
      ),
    );

    if (picked == null || !mounted) return;

    setState(() {
      _latitude = picked.latitude;
      _longitude = picked.longitude;
    });

    await _loadAddressFromLatLng(picked.latitude, picked.longitude);
  }

  Future<void> _submit() async {
    final valid = _formKey.currentState?.validate() ?? false;

    if (!valid) return;

    if (_latitude == null || _longitude == null) {
      TopToast.show(
        context,
        'Please select location first.',
        type: ToastType.error,
      );
      return;
    }

    setState(() => _saving = true);

    try {
      final api = ApiService(authService: _authService);

      await api.connectBinFromQr(
        binId: widget.binId,
        name: _nameCtrl.text.trim(),
        locationDescription: _descCtrl.text.trim(),
        latitude: _latitude!,
        longitude: _longitude!,
      );

      if (!mounted) return;

      Navigator.pop(context, true);
    } catch (e) {
      if (!mounted) return;

      var message = 'Failed to connect bin.';

      if (e is DioException) {
        message = e.response?.data?.toString() ??
            e.message ??
            'Failed to connect bin.';
      } else {
        message = e.toString();
      }

      TopToast.show(
        context,
        message,
        type: ToastType.error,
      );
    } finally {
      if (mounted) {
        setState(() => _saving = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final hasLocation = _latitude != null && _longitude != null;

    return Scaffold(
      backgroundColor: bgColor,
      body: SafeArea(
        child: Form(
          key: _formKey,
          child: SingleChildScrollView(
            padding: const EdgeInsets.fromLTRB(18, 14, 18, 30),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _Header(),

                const SizedBox(height: 28),

                _QrInfoCard(binId: widget.binId),

                const SizedBox(height: 24),

                _FieldLabel('Bin Name'),

                const SizedBox(height: 8),

                TextFormField(
                  controller: _nameCtrl,
                  onTapOutside: (_) {
                    FocusManager.instance.primaryFocus?.unfocus();
                  },
                  decoration: InputDecoration(
                    prefixIcon: const Icon(
                      Icons.delete_outline,
                      color: borderGreen,
                    ),
                    suffixIcon: const Icon(
                      Icons.check_circle_outline,
                      color: borderGreen,
                    ),
                    hintText: 'Enter bin name',
                    filled: true,
                    fillColor: Colors.white,
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 16,
                      vertical: 18,
                    ),
                    enabledBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(16),
                      borderSide: const BorderSide(color: borderGreen),
                    ),
                    focusedBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(16),
                      borderSide: const BorderSide(
                        color: darkGreen,
                        width: 1.5,
                      ),
                    ),
                    errorBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(16),
                      borderSide: const BorderSide(color: Colors.red),
                    ),
                    focusedErrorBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(16),
                      borderSide: const BorderSide(color: Colors.red),
                    ),
                  ),
                  validator: (v) {
                    if (v == null || v.trim().isEmpty) {
                      return 'Please enter bin name';
                    }
                    return null;
                  },
                ),

                const SizedBox(height: 20),

                _FieldLabel('Location Description'),

                const SizedBox(height: 8),

                TextFormField(
                  controller: _descCtrl,
                  onTapOutside: (_) {
                    FocusManager.instance.primaryFocus?.unfocus();
                  },
                  minLines: 3,
                  maxLines: 4,
                  maxLength: 120,
                  decoration: InputDecoration(
                    prefixIcon: const Padding(
                      padding: EdgeInsets.only(bottom: 54),
                      child: Icon(
                        Icons.location_on_outlined,
                        color: borderGreen,
                      ),
                    ),
                    hintText: 'Enter location description',
                    filled: true,
                    fillColor: Colors.white,
                    counterStyle: const TextStyle(color: Colors.black54),
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 16,
                      vertical: 18,
                    ),
                    enabledBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(16),
                      borderSide: const BorderSide(color: borderGreen),
                    ),
                    focusedBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(16),
                      borderSide: const BorderSide(
                        color: darkGreen,
                        width: 1.5,
                      ),
                    ),
                  ),
                ),

                const SizedBox(height: 22),

                _PositionCard(
                  latitude: _latitude,
                  longitude: _longitude,
                  addressText: _addressText,
                  isResolvingAddress: _isResolvingAddress,
                  hasLocation: hasLocation,
                  saving: _saving,
                  onUseCurrentLocation: _useCurrentLocation,
                  onPickOnMap: _pickOnMap,
                ),

                const SizedBox(height: 28),

                SizedBox(
                  width: double.infinity,
                  height: 58,
                  child: ElevatedButton.icon(
                    onPressed: _saving ? null : _submit,
                    icon: _saving
                        ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: Colors.white,
                      ),
                    )
                        : const Icon(Icons.link),
                    label: Text(
                      _saving ? 'Connecting...' : 'Complete Connection',
                      style: const TextStyle(
                        fontSize: 17,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: darkGreen,
                      foregroundColor: Colors.white,
                      disabledBackgroundColor: darkGreen.withOpacity(0.45),
                      elevation: 0,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(18),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _Header extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        IconButton(
          onPressed: () {
            FocusManager.instance.primaryFocus?.unfocus();
            Navigator.pop(context);
          },
          icon: const Icon(
            Icons.arrow_back,
            size: 28,
          ),
        ),
        const SizedBox(width: 8),
        const Expanded(
          child: Text(
            'Connect New Bin',
            style: TextStyle(
              fontSize: 29,
              fontWeight: FontWeight.w500,
              letterSpacing: 0.2,
            ),
          ),
        ),
        Container(
          width: 44,
          height: 44,
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(14),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.06),
                blurRadius: 12,
                offset: const Offset(0, 6),
              ),
            ],
          ),
          child: const Icon(
            Icons.eco_outlined,
            color: Color(0xFF2F6B3D),
          ),
        ),
      ],
    );
  }
}

class _QrInfoCard extends StatelessWidget {
  const _QrInfoCard({
    required this.binId,
  });

  final String binId;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(18, 18, 18, 18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(
          color: const Color(0xFFE5EEDC),
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.06),
            blurRadius: 18,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            width: 58,
            height: 58,
            decoration: BoxDecoration(
              color: const Color(0xFFEAF6C8),
              borderRadius: BorderRadius.circular(16),
            ),
            child: const Icon(
              Icons.qr_code_2,
              color: Color(0xFF4CAF50),
              size: 36,
            ),
          ),

          const SizedBox(width: 16),

          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Bin ID from QR',
                  style: TextStyle(
                    fontSize: 17,
                    fontWeight: FontWeight.w800,
                    color: Color(0xFF2F6B3D),
                  ),
                ),

                const SizedBox(height: 8),

                Text(
                  binId,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 1,
                    color: Colors.black87,
                  ),
                ),

                const SizedBox(height: 8),

                const Text(
                  'ID is auto-read from QR, users do not need to type it.',
                  style: TextStyle(
                    fontSize: 13.5,
                    color: Colors.black54,
                    height: 1.35,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _FieldLabel extends StatelessWidget {
  const _FieldLabel(this.text);

  final String text;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(left: 2),
      child: Text(
        text,
        style: const TextStyle(
          fontSize: 16,
          fontWeight: FontWeight.w800,
          color: Color(0xFF12351E),
        ),
      ),
    );
  }
}

class _PositionCard extends StatelessWidget {
  const _PositionCard({
    required this.latitude,
    required this.longitude,
    required this.addressText,
    required this.isResolvingAddress,
    required this.hasLocation,
    required this.saving,
    required this.onUseCurrentLocation,
    required this.onPickOnMap,
  });

  final double? latitude;
  final double? longitude;
  final String? addressText;
  final bool isResolvingAddress;
  final bool hasLocation;
  final bool saving;
  final VoidCallback onUseCurrentLocation;
  final VoidCallback onPickOnMap;

  static const darkGreen = Color(0xFF0B5D1E);

  @override
  Widget build(BuildContext context) {
    final LatLng? point = hasLocation ? LatLng(latitude!, longitude!) : null;

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(22),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.07),
            blurRadius: 18,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(
                Icons.location_on_outlined,
                color: Colors.black87,
                size: 28,
              ),
              const SizedBox(width: 12),
              const Text(
                'Bin Position',
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.w900,
                  letterSpacing: 0.8,
                ),
              ),
              const Spacer(),
              if (hasLocation)
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 10,
                    vertical: 5,
                  ),
                  decoration: BoxDecoration(
                    color: const Color(0xFFEAF6C8),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: const Text(
                    'Selected',
                    style: TextStyle(
                      color: darkGreen,
                      fontSize: 12,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                ),
            ],
          ),

          const SizedBox(height: 10),

          Text(
            !hasLocation
                ? 'No location selected yet'
                : isResolvingAddress
                ? 'Finding address...'
                : addressText ?? 'Selected location',
            style: const TextStyle(
              fontSize: 14.5,
              color: Colors.black54,
              fontWeight: FontWeight.w600,
            ),
          ),

          const SizedBox(height: 14),

          ClipRRect(
            borderRadius: BorderRadius.circular(18),
            child: SizedBox(
              height: 145,
              width: double.infinity,
              child: hasLocation
                  ? FlutterMap(
                key: ValueKey(
                  '${point!.latitude}_${point.longitude}',
                ),
                options: MapOptions(
                  initialCenter: point,
                  initialZoom: 14.5,
                  interactionOptions: const InteractionOptions(
                    flags: InteractiveFlag.none,
                  ),
                ),
                children: [
                  TileLayer(
                    urlTemplate:
                    'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                    userAgentPackageName: 'com.example.pbl5Flutter',
                  ),
                  CircleLayer(
                    circles: [
                      CircleMarker(
                        point: point,
                        radius: 45,
                        color: const Color(0xFF4CAF50).withOpacity(0.18),
                        borderStrokeWidth: 0,
                      ),
                    ],
                  ),
                  MarkerLayer(
                    markers: [
                      Marker(
                        point: point,
                        width: 52,
                        height: 52,
                        child: const Icon(
                          Icons.location_pin,
                          color: Color(0xFF2F9E44),
                          size: 48,
                        ),
                      ),
                    ],
                  ),
                ],
              )
                  : Container(
                decoration: BoxDecoration(
                  color: const Color(0xFFF1F5EF),
                  borderRadius: BorderRadius.circular(18),
                ),
                child: const Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        Icons.map_outlined,
                        color: Colors.black38,
                        size: 36,
                      ),
                      SizedBox(height: 8),
                      Text(
                        'Map preview will appear here',
                        style: TextStyle(
                          color: Colors.black45,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),

          const SizedBox(height: 16),

          _PositionButton(
            icon: Icons.my_location,
            label: 'Use current location',
            onPressed: saving ? null : onUseCurrentLocation,
          ),

          const SizedBox(height: 12),

          _PositionButton(
            icon: Icons.map_outlined,
            label: 'Pick on map',
            onPressed: saving ? null : onPickOnMap,
          ),
        ],
      ),
    );
  }
}

class _PositionButton extends StatelessWidget {
  const _PositionButton({
    required this.icon,
    required this.label,
    required this.onPressed,
  });

  final IconData icon;
  final String label;
  final VoidCallback? onPressed;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      height: 52,
      child: OutlinedButton.icon(
        onPressed: onPressed,
        icon: Icon(icon),
        label: Align(
          alignment: Alignment.centerLeft,
          child: Text(
            label,
            style: const TextStyle(
              fontSize: 15.5,
              fontWeight: FontWeight.w800,
            ),
          ),
        ),
        style: OutlinedButton.styleFrom(
          foregroundColor: const Color(0xFF2F6B3D),
          side: BorderSide(
            color: Colors.black.withOpacity(0.12),
          ),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(18),
          ),
          padding: const EdgeInsets.symmetric(horizontal: 18),
        ),
      ),
    );
  }
}
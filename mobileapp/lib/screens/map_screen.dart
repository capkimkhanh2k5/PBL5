import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:latlong2/latlong.dart';

class MapScreen extends StatefulWidget {
  final String? initialBinId;
  const MapScreen({super.key, this.initialBinId});

  @override
  State<MapScreen> createState() => MapScreenState();
}

class MapScreenState extends State<MapScreen>
    with TickerProviderStateMixin {
  final MapController _mapController = MapController();

  LatLng? _previousCenter;
  double? _previousZoom;
  bool _isZoomedToMarker = false;
  bool _isAnimating = false;
  String? _selectedBinId;
  bool _showBinList = false;

  late Future<List<Map<String, dynamic>>> _binsFuture;

  @override
  void initState() {
    super.initState();
    if (widget.initialBinId != null) {
      _selectedBinId = widget.initialBinId;
    }
    _binsFuture = fetchBins();
  }

  Future<List<Map<String, dynamic>>> fetchBins() async {
    final snapshot =
    await FirebaseFirestore.instance.collection('bins_metadata').get();

    return snapshot.docs.map((doc) {
      final data = doc.data();

      return {
        'id': doc.id,
        'name': data['name'] ?? 'Unknown Bin',
        'latitude': (data['latitude'] ?? 0).toDouble(),
        'longitude': (data['longitude'] ?? 0).toDouble(),
      };
    }).where((bin) {
      return bin['latitude'] != 0 && bin['longitude'] != 0;
    }).toList();
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
        await _animatedMove(
          target,
          12,
          durationMs: 700,
        );
      }

      await _animatedMove(
        target,
        16,
        durationMs: 1000,
      );
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
        durationMs: 1000,
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

  Widget _buildBinDropdown(List<Map<String, dynamic>> bins) {
    return Positioned(
      top: 16,
      right: 16,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Material(
            elevation: 6,
            borderRadius: BorderRadius.circular(16),
            child: InkWell(
              borderRadius: BorderRadius.circular(16),
              onTap: () {
                setState(() {
                  _showBinList = !_showBinList;
                });
              },
              child: Container(
                padding:
                const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(
                      Icons.delete_outline,
                      color: Color(0xFF2F6B3D),
                    ),
                    const SizedBox(width: 8),
                    const Text(
                      "SmartBin",
                      style: TextStyle(fontWeight: FontWeight.w600),
                    ),
                    const SizedBox(width: 6),
                    AnimatedRotation(
                      turns: _showBinList ? 0.5 : 0,
                      duration: const Duration(milliseconds: 200),
                      child: const Icon(Icons.expand_more),
                    ),
                  ],
                ),
              ),
            ),
          ),
          if (_showBinList)
            Container(
              margin: const EdgeInsets.only(top: 8),
              width: 220,
              constraints: const BoxConstraints(maxHeight: 280),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                boxShadow: const [
                  BoxShadow(color: Colors.black26, blurRadius: 8),
                ],
              ),
              child: ListView.builder(
                shrinkWrap: true,
                itemCount: bins.length,
                itemBuilder: (context, index) {
                  final bin = bins[index];

                  return ListTile(
                    title: Text(bin['name']),
                    onTap: () async {
                      setState(() => _showBinList = false);
                      await _zoomToBin(bin);
                    },
                  );
                },
              ),
            ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('SmartBin Map'),
      ),
      body: FutureBuilder<List<Map<String, dynamic>>>(
        future: _binsFuture,
        builder: (context, snapshot) {
          if (!snapshot.hasData) {
            return const Center(child: CircularProgressIndicator());
          }

          final bins = snapshot.data!;
          
          LatLng initCenter = const LatLng(14.5, 108.0);
          double initZoom = 6.0;
          
          if (widget.initialBinId != null && _selectedBinId == widget.initialBinId) {
            try {
              final targetBin = bins.firstWhere((b) => b['id'] == widget.initialBinId);
              initCenter = LatLng(targetBin['latitude'], targetBin['longitude']);
              initZoom = 16.0;
            } catch (_) {}
          }

          return Stack(
            children: [
              FlutterMap(
                mapController: _mapController,
                options: MapOptions(
                  initialCenter: initCenter,
                  initialZoom: initZoom,
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
                  MarkerLayer(
                    markers: bins.map((bin) {
                      final isSelected = _selectedBinId == bin['id'];

                      return Marker(
                        point: LatLng(bin['latitude'], bin['longitude']),
                        width: 150,
                        height: 90,
                        child: GestureDetector(
                          onTap: () => _zoomToBin(bin),
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              if (isSelected)
                                Container(
                                  padding: const EdgeInsets.symmetric(
                                    horizontal: 10,
                                    vertical: 5,
                                  ),
                                  margin: const EdgeInsets.only(bottom: 4),
                                  decoration: BoxDecoration(
                                    color: Colors.white,
                                    borderRadius: BorderRadius.circular(8),
                                    boxShadow: const [
                                      BoxShadow(
                                        color: Colors.black26,
                                        blurRadius: 4,
                                      ),
                                    ],
                                  ),
                                  child: Text(
                                    bin['name'],
                                    maxLines: 1,
                                    overflow: TextOverflow.ellipsis,
                                    style: const TextStyle(
                                      fontSize: 12,
                                      fontWeight: FontWeight.w600,
                                    ),
                                  ),
                                ),
                              Icon(
                                Icons.location_on,
                                size: isSelected ? 45 : 40,
                                color: Colors.green,
                              ),
                            ],
                          ),
                        ),
                      );
                    }).toList(),
                  ),
                ],
              ),
              _buildBinDropdown(bins),
            ],
          );
        },
      ),
    );
  }
}
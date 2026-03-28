import 'dart:convert';

import 'package:flutter/material.dart';
import '../utils/top_toast.dart';
import 'package:mobile_scanner/mobile_scanner.dart';

import 'bin_connection_form_screen.dart';

class ScanQrScreen extends StatefulWidget {
  const ScanQrScreen({super.key});

  @override
  State<ScanQrScreen> createState() => _ScanQrScreenState();
}

class _ScanQrScreenState extends State<ScanQrScreen> {
  final MobileScannerController _controller = MobileScannerController(
    detectionSpeed: DetectionSpeed.noDuplicates,
  );

  bool _processing = false;

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> _onDetect(BarcodeCapture capture) async {
    if (_processing) return;

    final raw = capture.barcodes.isNotEmpty ? capture.barcodes.first.rawValue : null;
    if (raw == null || raw.trim().isEmpty) return;

    final binId = _extractBinId(raw.trim());
    if (binId == null || binId.isEmpty) {
      if (!mounted) return;
      TopToast.show(context, 'Invalid QR data. Cannot read binId.');
      return;
    }

    _processing = true;
    await _controller.stop();

    if (!mounted) return;

    final connected = await Navigator.push<bool>(
      context,
      MaterialPageRoute(
        builder: (_) => BinConnectionFormScreen(binId: binId),
      ),
    );

    if (!mounted) return;

    if (connected == true) {
      Navigator.pop(context, true);
      return;
    }

    _processing = false;
    await _controller.start();
  }

  String? _extractBinId(String raw) {
    if (raw.startsWith('{') && raw.endsWith('}')) {
      try {
        final decoded = jsonDecode(raw);
        if (decoded is Map<String, dynamic>) {
          final v = decoded['binId'] ?? decoded['id'] ?? decoded['bin_id'];
          if (v != null) return v.toString();
        }
      } catch (_) {
        // fallback to other parser
      }
    }

    final uri = Uri.tryParse(raw);
    if (uri != null) {
      final fromQuery = uri.queryParameters['binId'] ??
          uri.queryParameters['id'] ??
          uri.queryParameters['bin_id'];
      if (fromQuery != null && fromQuery.trim().isNotEmpty) {
        return fromQuery.trim();
      }

      final lastPath = uri.pathSegments.isNotEmpty ? uri.pathSegments.last : '';
      if (lastPath.isNotEmpty) return lastPath;
    }

    return raw;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Scan QR'),
        actions: [
          IconButton(
            onPressed: () => _controller.toggleTorch(),
            icon: const Icon(Icons.flash_on),
          ),
        ],
      ),
      body: Stack(
        children: [
          MobileScanner(
            controller: _controller,
            onDetect: _onDetect,
          ),
          Align(
            alignment: Alignment.bottomCenter,
            child: Container(
              margin: const EdgeInsets.all(16),
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.black.withValues(alpha: 0.6),
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Text(
                'Point camera to QR code on the bin to start connection.',
                style: TextStyle(color: Colors.white),
                textAlign: TextAlign.center,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
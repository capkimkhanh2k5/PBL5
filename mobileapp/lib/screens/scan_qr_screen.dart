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
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Invalid QR data. This is not a valid Smart Bin QR code.')),
      );
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
          // 1. Xác minh hệ thống
          if (decoded['system'] != 'PBL5-SmartBin') {
            return null; // Từ chối QR không có chữ ký của PBL5
          }
          
          // 2. Lấy ID thùng rác
          final v = decoded['binId'] ?? decoded['id'] ?? decoded['bin_id'];
          if (v != null && v.toString().trim().isNotEmpty) {
            return v.toString().trim();
          }
        }
      } catch (_) {
        // Lỗi parse JSON
        return null;
      }
    }

    // Nếu không khớp chuẩn JSON hoặc bất kỳ lý do gì, luôn từ chối (trả về null)
    return null;
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
                color: Colors.black.withOpacity(0.6),
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
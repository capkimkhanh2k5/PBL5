import 'package:flutter/material.dart';
import '../utils/top_toast.dart';
import 'history_screen.dart';
import 'map_screen.dart';
import '../services/api_service.dart';
import '../services/auth_service.dart';

class BinDetailScreen extends StatefulWidget {
  const BinDetailScreen({super.key, required this.binId});
  final String binId;

  @override
  State<BinDetailScreen> createState() => _BinDetailScreenState();
}

class _BinDetailScreenState extends State<BinDetailScreen> {
  final _authService = AuthService();
  _BinDetail? data;
  bool _isLoading = true;
  String? _error;

  static const bg = Color(0xFFEAF6EE);
  static const primary = Color(0xFF2F6B3D);

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final api = ApiService(authService: _authService);
      final results = await Future.wait([
        api.getAllBinStatuses(),
        api.getRecentSensorLogs(widget.binId, limit: 24),
      ]);
      final statuses = results[0];
      final status = statuses.cast<Map<String, dynamic>?>().firstWhere(
            (e) => (e?['id'] ?? '').toString() == widget.binId,
            orElse: () => null,
          );
      final logs = results[1];

      int fillOrganic = _toInt(status?['fillOrganic']) ?? 0;
      int fillRecycle = _toInt(status?['fillRecycle']) ?? 0;
      int fillNonRecycle = _toInt(status?['fillNonRecycle']) ?? 0;
      int fillHazardous = _toInt(status?['fillHazardous']) ?? 0;

      if (status == null && logs.isNotEmpty) {
        final latest = logs.first;
        fillOrganic = _toInt(latest['fillOrganic']) ?? fillOrganic;
        fillRecycle = _toInt(latest['fillRecycle']) ?? fillRecycle;
        fillNonRecycle = _toInt(latest['fillNonRecycle']) ?? fillNonRecycle;
        fillHazardous = _toInt(latest['fillHazardous']) ?? fillHazardous;
      }

      final values = [fillOrganic, fillRecycle, fillNonRecycle, fillHazardous];
      final avgFill = values.reduce((a, b) => a + b) ~/ values.length;

      double etaDays = 0;
      if (logs.length >= 2) {
        final newest = logs.first;
        final oldest = logs.last;
        final newestAvg = _avgFromLog(newest);
        final oldestAvg = _avgFromLog(oldest);
        final newestAt = _toInt(newest['recordedAt']);
        final oldestAt = _toInt(oldest['recordedAt']);
        if (newestAt != null && oldestAt != null && newestAt > oldestAt) {
          final hours = (newestAt - oldestAt) / 3600000.0;
          final delta = newestAvg - oldestAvg;
          if (hours > 0 && delta > 0) {
            final ratePerHour = delta / hours;
            etaDays = ((100 - avgFill) / ratePerHour) / 24.0;
          }
        }
      }
      if (etaDays <= 0 || etaDays.isNaN || etaDays.isInfinite) {
        etaDays = ((100 - avgFill) / 15.0).clamp(0.2, 14.0).toDouble();
      }

      if (!mounted) return;
      setState(() {
        data = _BinDetail(
          fillPercent: avgFill,
          suggestDumpAt: 90,
          etaDays: etaDays,
          fillOrganic: fillOrganic,
          fillRecycle: fillRecycle,
          fillNonRecycle: fillNonRecycle,
          fillHazardous: fillHazardous,
        );
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = 'Failed to load bin detail from backend.');
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  static int? _toInt(dynamic value) {
    if (value is int) return value;
    if (value is num) return value.toInt();
    return int.tryParse(value?.toString() ?? '');
  }

  static int _avgFromLog(Map<String, dynamic> log) {
    final values = [
      _toInt(log['fillOrganic']) ?? 0,
      _toInt(log['fillRecycle']) ?? 0,
      _toInt(log['fillNonRecycle']) ?? 0,
      _toInt(log['fillHazardous']) ?? 0,
    ];
    return values.reduce((a, b) => a + b) ~/ values.length;
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    if (_error != null || data == null) {
      return Scaffold(
        appBar: AppBar(title: Text(widget.binId)),
        body: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(_error ?? 'Cannot load data'),
              const SizedBox(height: 12),
              ElevatedButton(onPressed: _loadData, child: const Text('Retry')),
            ],
          ),
        ),
      );
    }

    final d = data!;

    return Scaffold(
      backgroundColor: bg,
      appBar: AppBar(
        backgroundColor: bg,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.black),
          onPressed: () => Navigator.pop(context),
        ),
        title: Text(
          widget.binId,
          style: const TextStyle(
            color: Colors.black,
            fontWeight: FontWeight.w800,
            fontSize: 26,
          ),
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
        child: Column(
          children: [
            _softCard(
              child: Column(
                children: [
                  const SizedBox(height: 10),
                  const Text(
                    
                    "Bin Fill Level",
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.w800,
                      color: Colors.black87,
                    ),
                  ),
                  const SizedBox(height: 6),

                  

                  _gauge(d.fillPercent),


                  

                  const Align(
                    alignment: Alignment.centerLeft,
                    child: Text(
                      "Waste Composition",
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                  

                  GridView.count(
                    crossAxisCount: 2,
                    crossAxisSpacing: 12,
                    mainAxisSpacing: 12,
                    childAspectRatio: 1.8,
                    shrinkWrap: true,
                    physics: const NeverScrollableScrollPhysics(),
                    children: [
                      WasteMiniCard(
                        title: "Organic Waste",
                        percent: d.fillOrganic / 100,
                        color: const Color(0xFF2D8CFF),
                      ),
                      WasteMiniCard(
                        title: "Plastic & Paper",
                        percent: d.fillRecycle / 100,
                        color: const Color(0xFFF6C000),
                      ),
                      WasteMiniCard(
                        title: "Metal",
                        percent: d.fillNonRecycle / 100,
                        color: const Color(0xFFFF8A00),
                      ),
                      WasteMiniCard(
                        title: "Other Waste",
                        percent: d.fillHazardous / 100,
                        color: const Color(0xFFFF3B30),
                      ),
                    ],
                  ),
                  
                  const SizedBox(height: 14),
                  _primaryButton(
                    text: 'View History',
                    onTap: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (_) => HistoryScreen(binId: widget.binId),
                        ),
                      );
                    },
                  ),
                ],
              ),
            ),
            const SizedBox(height: 14),
            _softCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Suggestion',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w800,
                      color: Colors.black87,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Recommended to empty at: ${d.suggestDumpAt}%',
                    style: const TextStyle(fontSize: 14, color: Colors.black87),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Estimated to be full in: ~ ${d.etaDays.toStringAsFixed(1)} ngày',
                    style: const TextStyle(fontSize: 14, color: Colors.black87),
                  ),
                  const SizedBox(height: 14),
                  _primaryButton(
                    text: 'View Bin Location',
                    onTap: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (_) => MapScreen(
                            initialBinId: widget.binId,
                          ),
                        ),
                      );
                    },
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }


  Widget _softCard({required Widget child}) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.06),
            blurRadius: 12,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: child,
    );
  }

  Widget _primaryButton({required String text, required VoidCallback onTap}) {
    return SizedBox(
      width: double.infinity,
      height: 46,
      child: ElevatedButton(
        style: ElevatedButton.styleFrom(
          backgroundColor: primary,
          foregroundColor: Colors.white,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(24),
          ),
          elevation: 0,
        ),
        onPressed: onTap,
        child: Text(
          text,
          style: const TextStyle(fontWeight: FontWeight.w700),
        ),
      ),
    );
  }



  Widget _gauge(int percent) {
    return SizedBox(
      height: 220, // chỉnh 200-240 tùy bạn
      child: Center(
        child: Stack(
          alignment: Alignment.center,
          children: [
            CustomPaint(
              size: const Size(180, 180),
              painter: MultiColorRingPainter(percent),
            ),
            Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  '$percent%',
                  style: const TextStyle(
                    fontSize: 40,
                    fontWeight: FontWeight.w800,
                    color: Colors.black87,
                  ),
                ),
                const SizedBox(height: 2),
                const Text(
                  'Fill Level',
                  style: TextStyle(color: Colors.black54),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
class WasteMiniCard extends StatelessWidget {
  const WasteMiniCard({
    super.key,
    required this.title,
    required this.percent,
    required this.color,
  });

  final String title;
  final double percent;
  final Color color;

  @override
  Widget build(BuildContext context) {
    final pct = (percent * 100).round();

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.06),
            blurRadius: 14,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Stack(
        children: [
          // “dấu màu nhỏ ở góc” giống ảnh
          Positioned(
            top: 2,
            right: 2,
            child: Container(
              width: 10,
              height: 10,
              decoration: BoxDecoration(color: color, shape: BoxShape.circle),
            ),
          ),

          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w800,
                  color: Colors.black87,
                ),
              ),
              const SizedBox(height: 6),

              ClipRRect(
                borderRadius: BorderRadius.circular(999),
                child: LinearProgressIndicator(
                  value: percent.clamp(0.0, 1.0),
                  minHeight: 6,
                  backgroundColor: Colors.grey.shade200,
                  valueColor: AlwaysStoppedAnimation(color),
                ),
              ),
              const SizedBox(height: 6),

              Align(
                alignment: Alignment.centerRight,
                child: Text(
                  "$pct%",
                  style: const TextStyle(
                    fontSize: 11.5,
                    fontWeight: FontWeight.w800,
                    color: Colors.black87,
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

/* =======================
   Fake repo + data model
   ======================= */

class _BinDetail {
  final int fillPercent;
  final int suggestDumpAt;
  final double etaDays;
  final int fillOrganic;
  final int fillRecycle;
  final int fillNonRecycle;
  final int fillHazardous;

  const _BinDetail({
    required this.fillPercent,
    required this.suggestDumpAt,
    required this.etaDays,
    required this.fillOrganic,
    required this.fillRecycle,
    required this.fillNonRecycle,
    required this.fillHazardous,
  });
}

class MultiColorRingPainter extends CustomPainter {
  final int percent;
  MultiColorRingPainter(this.percent);

  @override
  void paint(Canvas canvas, Size size) {
    final strokeWidth = 14.0;
    final center = Offset(size.width / 2, size.height / 2);
    final radius = (size.width / 2) - strokeWidth;

    final rect = Rect.fromCircle(center: center, radius: radius);

    final bgPaint = Paint()
      ..color = const Color(0xFFE6E6E6)
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round;

    canvas.drawArc(rect, 0, 6.28, false, bgPaint);

    final colors = [
      const Color(0xFF2D8CFF), // Rác 1
      const Color(0xFFF6C000), // Rác 2
      const Color(0xFFFF8A00), // Rác 3
      const Color(0xFFFF3B30), // Rác 4
    ];

    final sweepAngle = 6.28318 * (percent / 100);

    double startAngle = -1.57; // bắt đầu từ trên

    for (int i = 0; i < colors.length; i++) {
      final paint = Paint()
        ..color = colors[i]
        ..style = PaintingStyle.stroke
        ..strokeWidth = strokeWidth
        ..strokeCap = StrokeCap.round;

      final segment = sweepAngle / 4;

      canvas.drawArc(
        rect,
        startAngle,
        segment,
        false,
        paint,
      );

      startAngle += segment;
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
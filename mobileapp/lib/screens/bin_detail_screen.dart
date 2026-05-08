import 'package:flutter/material.dart';
import '../utils/top_toast.dart';
import 'history_screen.dart';
import 'map_screen.dart';
import '../services/api_service.dart';
import '../services/auth_service.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

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
  DateTime _selectedChartDate = DateTime.now();

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
        api.getRecentSensorLogs(widget.binId, limit: 288),
      ]);

      final statuses = results[0];
      final status = statuses.cast<Map<String, dynamic>?>().firstWhere(
            (e) => (e?['id'] ?? '').toString() == widget.binId,
        orElse: () => null,
      );

      final logs = results[1];

      print('STATUS = $status');
      print('FIRST LOG = ${logs.isNotEmpty ? logs.first : 'EMPTY'}');
      print('BIN ID = ${widget.binId}');

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
        final newestAt = FillLineChart._toMillis(newest['recordedAt']);
        final oldestAt = FillLineChart._toMillis(oldest['recordedAt']);

        if (newestAt > 0 && oldestAt > 0 && newestAt > oldestAt) {
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

      final newestLogDate = logs.isNotEmpty
          ? DateTime.fromMillisecondsSinceEpoch(
        FillLineChart._toMillis(logs.first['recordedAt']),
      )
          : DateTime.now();

      setState(() {
        _selectedChartDate = newestLogDate;
        data = _BinDetail(
          fillPercent: avgFill,
          suggestDumpAt: 90,
          etaDays: etaDays,
          fillOrganic: fillOrganic,
          fillRecycle: fillRecycle,
          fillNonRecycle: fillNonRecycle,
          fillHazardous: fillHazardous,
          logs: logs.cast<Map<String, dynamic>>(),
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
    final chartLogs = _filterLogsByDate(d.logs, _selectedChartDate);

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
                    'Bin Fill Level',
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
                      'Waste Composition',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                  GridView.count(
                    crossAxisCount: 2,
                    crossAxisSpacing: 12,
                    mainAxisSpacing: 12,
                    childAspectRatio: 1.8,
                    shrinkWrap: true,
                    physics: const NeverScrollableScrollPhysics(),
                    children: [
                      WasteMiniCard(
                        title: 'Organic Waste',
                        percent: d.fillOrganic / 100,
                        color: const Color(0xFF2D8CFF),
                      ),
                      WasteMiniCard(
                        title: 'Plastic & Paper',
                        percent: d.fillRecycle / 100,
                        color: const Color(0xFFF6C000),
                      ),
                      WasteMiniCard(
                        title: 'Metal',
                        percent: d.fillNonRecycle / 100,
                        color: const Color(0xFFFF8A00),
                      ),
                      WasteMiniCard(
                        title: 'Other Waste',
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
            const SizedBox(height: 14),
            _softCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Today Fill Analytics',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w800,
                      color: Colors.black87,
                    ),
                  ),
                  const SizedBox(height: 4),
                  const Text(
                    'Fill level trend during the day',
                    style: TextStyle(
                      fontSize: 13,
                      color: Colors.black54,
                    ),
                  ),
                  const SizedBox(height: 12),
                  _datePickerButton(),
                  const SizedBox(height: 18),
                  SizedBox(
                    height: 220,
                    child: FillLineChart(logs: chartLogs),
                  ),
                  const SizedBox(height: 14),
                  const Wrap(
                    spacing: 14,
                    runSpacing: 8,
                    children: [
                      _LegendDot(
                        label: 'Organic',
                        color: Color(0xFF2D8CFF),
                      ),
                      _LegendDot(
                        label: 'Recycle',
                        color: Color(0xFFF6C000),
                      ),
                      _LegendDot(
                        label: 'NonRecycle',
                        color: Color(0xFFFF8A00),
                      ),
                      _LegendDot(
                        label: 'Hazardous',
                        color: Color(0xFFFF3B30),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  List<Map<String, dynamic>> _filterLogsByDate(
      List<Map<String, dynamic>> logs,
      DateTime selectedDate,
      ) {
    return logs.where((log) {
      final millis = FillLineChart._toMillis(log['recordedAt']);
      if (millis <= 0) return false;

      final dt = DateTime.fromMillisecondsSinceEpoch(millis);
      return dt.year == selectedDate.year &&
          dt.month == selectedDate.month &&
          dt.day == selectedDate.day;
    }).toList();
  }

  String _formatSelectedDate(DateTime date) {
    final day = date.day.toString().padLeft(2, '0');
    final month = date.month.toString().padLeft(2, '0');
    final year = date.year.toString();
    return '$day/$month/$year';
  }

  Future<void> _pickChartDate() async {
    final picked = await showDatePicker(
      context: context,
      initialDate: _selectedChartDate,
      firstDate: DateTime(2020),
      lastDate: DateTime(2035),
    );

    if (picked == null) return;

    setState(() {
      _selectedChartDate = picked;
    });
  }

  Widget _datePickerButton() {
    return InkWell(
      borderRadius: BorderRadius.circular(14),
      onTap: _pickChartDate,
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
        decoration: BoxDecoration(
          color: const Color(0xFFF4F7F5),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: Colors.black.withOpacity(0.06)),
        ),
        child: Row(
          children: [
            const Icon(
              Icons.calendar_today_rounded,
              size: 18,
              color: primary,
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                'Date: ${_formatSelectedDate(_selectedChartDate)}',
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w700,
                  color: Colors.black87,
                ),
              ),
            ),
            const Icon(
              Icons.keyboard_arrow_down_rounded,
              color: Colors.black54,
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
      height: 220,
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
                  '$pct%',
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

class _LegendDot extends StatelessWidget {
  const _LegendDot({
    required this.label,
    required this.color,
  });

  final String label;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 9,
          height: 9,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
        const SizedBox(width: 5),
        Text(
          label,
          style: const TextStyle(fontSize: 11.5, fontWeight: FontWeight.w600),
        ),
      ],
    );
  }
}

class FillLineChart extends StatelessWidget {
  const FillLineChart({super.key, required this.logs});

  final List<Map<String, dynamic>> logs;

  static int _toMillis(dynamic value) {
    if (value == null) return 0;

    if (value is Timestamp) {
      return value.millisecondsSinceEpoch;
    }

    if (value is DateTime) {
      return value.millisecondsSinceEpoch;
    }

    if (value is int) {
      // Trường hợp API trả numeric timestamp.
      // Nếu là seconds thì đổi sang milliseconds.
      if (value < 1000000000000) return value * 1000;
      return value;
    }

    if (value is num) {
      final intValue = value.toInt();
      if (intValue < 1000000000000) return intValue * 1000;
      return intValue;
    }

    // Trường hợp Timestamp bị convert thành map/json.
    if (value is Map) {
      final seconds = value['seconds'] ?? value['_seconds'];
      final nanos = value['nanoseconds'] ?? value['_nanoseconds'] ?? 0;

      final secInt = int.tryParse(seconds?.toString() ?? '');
      final nanoInt = int.tryParse(nanos?.toString() ?? '') ?? 0;

      if (secInt != null) {
        return secInt * 1000 + (nanoInt ~/ 1000000);
      }
    }

    final parsedInt = int.tryParse(value.toString());
    if (parsedInt != null) {
      if (parsedInt < 1000000000000) return parsedInt * 1000;
      return parsedInt;
    }

    final parsedDate = DateTime.tryParse(value.toString());
    if (parsedDate != null) {
      return parsedDate.millisecondsSinceEpoch;
    }

    return 0;
  }

  static List<Map<String, dynamic>> _sampleLogs(
      List<Map<String, dynamic>> logs,
      int count,
      ) {
    if (logs.length <= count) return logs;

    return List.generate(count, (i) {
      final index = (i * (logs.length - 1) / (count - 1)).round();
      return logs[index];
    });
  }

  @override
  Widget build(BuildContext context) {
    final sortedLogs = [...logs];

    sortedLogs.sort((a, b) {
      final at = _toMillis(a['recordedAt']);
      final bt = _toMillis(b['recordedAt']);
      return at.compareTo(bt);
    });

    final validLogs = sortedLogs.where((log) {
      return _toMillis(log['recordedAt']) > 0;
    }).toList();

    final points = validLogs.length > 48 ? _sampleLogs(validLogs, 48) : validLogs;

    if (points.length < 2) {
      return const Center(
        child: Text(
          'Not enough data to display chart',
          style: TextStyle(color: Colors.black45),
        ),
      );
    }

    return CustomPaint(
      painter: FillLineChartPainter(points),
      child: Container(),
    );
  }
}

class FillLineChartPainter extends CustomPainter {
  FillLineChartPainter(this.logs);

  final List<Map<String, dynamic>> logs;

  static int _toInt(dynamic value) {
    if (value is int) return value;
    if (value is num) return value.toInt();
    return int.tryParse(value?.toString() ?? '') ?? 0;
  }

  static String _formatTimeLabel(int millis) {
    final dt = DateTime.fromMillisecondsSinceEpoch(millis);
    final hour = dt.hour.toString().padLeft(2, '0');
    final minute = dt.minute.toString().padLeft(2, '0');
    return '$hour:$minute';
  }

  @override
  void paint(Canvas canvas, Size size) {
    const left = 38.0;
    const right = 10.0;
    const top = 12.0;
    const bottom = 32.0;

    final chartW = size.width - left - right;
    final chartH = size.height - top - bottom;

    final firstTime = FillLineChart._toMillis(logs.first['recordedAt']);
    final lastTime = FillLineChart._toMillis(logs.last['recordedAt']);
    final totalDuration = (lastTime - firstTime).toDouble();

    final gridPaint = Paint()
      ..color = Colors.grey.withOpacity(0.18)
      ..strokeWidth = 1;

    final textPainter = TextPainter(textDirection: TextDirection.ltr);

    // Trục Y: 100, 75, 50, 25, 0
    for (int i = 0; i <= 4; i++) {
      final y = top + chartH * i / 4;
      canvas.drawLine(Offset(left, y), Offset(size.width - right, y), gridPaint);

      final label = '${100 - i * 25}';
      textPainter.text = TextSpan(
        text: label,
        style: const TextStyle(fontSize: 10, color: Colors.black45),
      );
      textPainter.layout();
      textPainter.paint(canvas, Offset(0, y - 6));
    }

    // Trục X theo thời gian thật của recordedAt, không hardcode 00h/06h nữa.
    for (int i = 0; i <= 4; i++) {
      final ratio = i / 4;
      final x = left + chartW * ratio;
      final labelMillis = firstTime + ((lastTime - firstTime) * ratio).toInt();
      final label = _formatTimeLabel(labelMillis);

      canvas.drawLine(Offset(x, top), Offset(x, top + chartH), gridPaint);

      textPainter.text = TextSpan(
        text: label,
        style: const TextStyle(fontSize: 10, color: Colors.black45),
      );
      textPainter.layout();
      textPainter.paint(
        canvas,
        Offset(x - textPainter.width / 2, top + chartH + 8),
      );
    }

    void drawLine(String key, Color color) {
      final path = Path();

      for (int i = 0; i < logs.length; i++) {
        final value = _toInt(logs[i][key]).clamp(0, 100);
        final currentTime = FillLineChart._toMillis(logs[i]['recordedAt']);

        final progress = totalDuration <= 0
            ? i / (logs.length - 1)
            : (currentTime - firstTime) / totalDuration;

        final x = left + chartW * progress.clamp(0.0, 1.0);
        final y = top + chartH * (1 - value / 100);

        if (i == 0) {
          path.moveTo(x, y);
        } else {
          path.lineTo(x, y);
        }
      }

      final paint = Paint()
        ..color = color
        ..strokeWidth = 2.6
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round
        ..strokeJoin = StrokeJoin.round;

      canvas.drawPath(path, paint);
    }

    drawLine('fillOrganic', const Color(0xFF2D8CFF));
    drawLine('fillRecycle', const Color(0xFFF6C000));
    drawLine('fillNonRecycle', const Color(0xFFFF8A00));
    drawLine('fillHazardous', const Color(0xFFFF3B30));
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}

class _BinDetail {
  final int fillPercent;
  final int suggestDumpAt;
  final double etaDays;
  final int fillOrganic;
  final int fillRecycle;
  final int fillNonRecycle;
  final int fillHazardous;
  final List<Map<String, dynamic>> logs;

  const _BinDetail({
    required this.fillPercent,
    required this.suggestDumpAt,
    required this.etaDays,
    required this.fillOrganic,
    required this.fillRecycle,
    required this.fillNonRecycle,
    required this.fillHazardous,
    required this.logs,
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
      const Color(0xFF2D8CFF),
      const Color(0xFFF6C000),
      const Color(0xFFFF8A00),
      const Color(0xFFFF3B30),
    ];

    final sweepAngle = 6.28318 * (percent / 100);

    double startAngle = -1.57;

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

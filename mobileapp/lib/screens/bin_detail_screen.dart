import 'package:flutter/material.dart';
import 'history_screen.dart';

class BinDetailScreen extends StatefulWidget {
  const BinDetailScreen({super.key, required this.binId});
  final String binId;

  @override
  State<BinDetailScreen> createState() => _BinDetailScreenState();
}

class _BinDetailScreenState extends State<BinDetailScreen> {
  late final _BinDetail data;

  static const bg = Color(0xFFEAF6EE);
  static const primary = Color(0xFF2F6B3D);

  @override
  void initState() {
    super.initState();
    data = FakeBinRepo.getDetail(widget.binId); // ✅ fake data theo ID
  }

  @override
  Widget build(BuildContext context) {
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
                    
                    "Mức đầy thùng rác",
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.w800,
                      color: Colors.black87,
                    ),
                  ),
                  const SizedBox(height: 6),

                  

                  _gauge(data.fillPercent),


                  

                  const Align(
                    alignment: Alignment.centerLeft,
                    child: Text(
                      "Thành phần rác",
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
                    childAspectRatio: 1.9,
                    shrinkWrap: true,
                    physics: const NeverScrollableScrollPhysics(),
                    children: const [
                      WasteMiniCard(
                        title: "Rác hữu cơ",
                        percent: 0.70,
                        color: Color(0xFF2D8CFF),
                      ),
                      WasteMiniCard(
                        title: "Nhựa & giấy",
                        percent: 0.45,
                        color: Color(0xFFF6C000),
                      ),
                      WasteMiniCard(
                        title: "Kim loại",
                        percent: 0.20,
                        color: Color(0xFFFF8A00),
                      ),
                      WasteMiniCard(
                        title: "Rác khác",
                        
                        percent: 0.55,
                        color: Color(0xFFFF3B30),
                      ),
                    ],
                  ),
                  
                  const SizedBox(height: 14),
                  _primaryButton(
                    text: 'Xem lịch sử',
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
                    'Gợi ý',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w800,
                      color: Colors.black87,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Nên đổ khi đạt: ${data.suggestDumpAt}%',
                    style: const TextStyle(fontSize: 14, color: Colors.black87),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Dự kiến đầy trong: ~ ${data.etaDays.toStringAsFixed(1)} ngày',
                    style: const TextStyle(fontSize: 14, color: Colors.black87),
                  ),
                  const SizedBox(height: 14),
                  _primaryButton(
                    text: 'Xem vị trí thùng',
                    onTap: () {
                      // TODO: làm sau
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(content: Text('TODO: Map')),
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
                  'Độ đầy',
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
      padding: const EdgeInsets.all(12),
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

  const _BinDetail({
    required this.fillPercent,
    required this.suggestDumpAt,
    required this.etaDays,
  });
}

class FakeBinRepo {
  // Fake data theo TC-xx (mỗi TC khác nhau)
  static _BinDetail getDetail(String binId) {
    const map = <String, _BinDetail>{
      'TC-01': _BinDetail(fillPercent: 72, suggestDumpAt: 90, etaDays: 1.8),
      'TC-02': _BinDetail(fillPercent: 42, suggestDumpAt: 85, etaDays: 3.2),
      'TC-03': _BinDetail(fillPercent: 90, suggestDumpAt: 90, etaDays: 0.3),
      'TC-04': _BinDetail(fillPercent: 15, suggestDumpAt: 80, etaDays: 6.0),
    };

    // nếu TC-05 trở đi chưa có thì random nhẹ theo hash
    if (map.containsKey(binId)) return map[binId]!;
    final n = binId.codeUnits.fold<int>(0, (a, b) => a + b);
    final p = 20 + (n % 75); // 20..94
    final eta = (100 - p) / 30.0; // 0.2..2.6
    return _BinDetail(fillPercent: p, suggestDumpAt: 90, etaDays: eta);
  }
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
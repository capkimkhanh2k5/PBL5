import 'package:flutter/material.dart';
import '../services/api_service.dart';
import '../services/auth_service.dart';

class ScheduleScreen extends StatefulWidget {
  const ScheduleScreen({super.key});

  @override
  State<ScheduleScreen> createState() => _ScheduleScreenState();
}

class _ScheduleScreenState extends State<ScheduleScreen> {
  final _authService = AuthService();
  late final ApiService _api;
  bool _loading = true;
  bool _triggering = false;
  bool _reminderEnabled = false;
  String? _error;
  List<_PickupItem> _items = const [];

  @override
  void initState() {
    super.initState();
    _api = ApiService(authService: _authService);
    _load();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final schedule = await _api.getPickupSchedule(limit: 40);

      final items = schedule.map((row) {
        final id = (row['binId'] ?? 'Unknown').toString();
        final avg = _toInt(row['avgFill']) ?? 0;
        final predictedMs = _toInt(row['predictedPickupAt'])?.toInt() ?? 0;
        final eta = predictedMs > 0
            ? DateTime.fromMillisecondsSinceEpoch(predictedMs)
            : DateTime.now().add(const Duration(days: 1));
        final priority = (row['priority'] ?? 'LOW').toString();
        return _PickupItem(binId: id, avgFill: avg, eta: eta, priority: priority);
      }).toList();

      if (!mounted) return;
      setState(() => _items = items);
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = 'Failed to load schedule from backend.');
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _triggerAndReload() async {
    if (_triggering) return;
    setState(() => _triggering = true);
    try {
      await _api.triggerAggregateSensorLogs();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Triggered aggregate. Refreshing schedule...')),
      );
      await Future<void>.delayed(const Duration(seconds: 2));
      await _load();
    } catch (_) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Cannot trigger aggregate right now.')),
      );
    } finally {
      if (mounted) setState(() => _triggering = false);
    }
  }

  void _toggleReminder() {
    if (_items.isEmpty) return;
    final next = _items.first;
    setState(() => _reminderEnabled = !_reminderEnabled);
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
          _reminderEnabled
              ? 'Reminder enabled for ${next.binId} at ${next.formatted}'
              : 'Reminder disabled',
        ),
      ),
    );
  }

  static int? _toInt(dynamic value) {
    if (value is int) return value;
    if (value is num) return value.toInt();
    return int.tryParse(value?.toString() ?? '');
  }

  static const bgColor = Color(0xFFEAF6EE);
  static const primary = Color(0xFF2F6B3D);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: bgColor,
      appBar: AppBar(
        backgroundColor: bgColor,
        elevation: 0,
        centerTitle: true,
        title: const Text(
          'Lịch Đổ Rác',
          style: TextStyle(
            color: Colors.black87,
            fontWeight: FontWeight.w700,
            fontSize: 22,
          ),
        ),
      ),
      body: RefreshIndicator(
        onRefresh: _load,
        child: SingleChildScrollView(
          physics: const AlwaysScrollableScrollPhysics(),
          padding: const EdgeInsets.all(16),
          child: _loading
            ? const Center(child: CircularProgressIndicator())
            : _error != null
                ? Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(_error!),
                        const SizedBox(height: 8),
                        ElevatedButton(onPressed: _load, child: const Text('Retry')),
                      ],
                    ),
                  )
                : _items.isEmpty
                    ? Center(
                        child: Padding(
                          padding: const EdgeInsets.only(top: 48),
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              const Icon(Icons.event_busy, size: 54, color: Colors.black54),
                              const SizedBox(height: 12),
                              const Text(
                                'Chưa có lịch thu gom',
                                style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700),
                              ),
                              const SizedBox(height: 8),
                              const Text(
                                'Kéo xuống để tải lại hoặc trigger aggregate để tạo lịch mới.',
                                textAlign: TextAlign.center,
                              ),
                              const SizedBox(height: 16),
                              ElevatedButton.icon(
                                onPressed: _triggering ? null : _triggerAndReload,
                                icon: _triggering
                                    ? const SizedBox(
                                        width: 14,
                                        height: 14,
                                        child: CircularProgressIndicator(strokeWidth: 2),
                                      )
                                    : const Icon(Icons.play_circle_fill),
                                label: Text(_triggering ? 'Đang tạo...' : 'Trigger Aggregate'),
                              ),
                            ],
                          ),
                        ),
                      )
                : Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (_items.isNotEmpty)
            /// NEXT PICKUP CARD
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(22),
                gradient: const LinearGradient(
                  colors: [
                    Color(0xFFE8F5E9),
                    Color(0xFFD7EDDA),
                  ],
                ),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.08),
                    blurRadius: 18,
                    offset: const Offset(0, 10),
                  ),
                ],
              ),
              child: Column(
                children: [
                  Row(
                    children: [
                      const Icon(Icons.calendar_month,
                          color: primary, size: 28),
                      const SizedBox(width: 12),
                      Text(
                        "Next Pickup • ${_items.isNotEmpty ? _items.first.binId : '-'}",
                        style: const TextStyle(
                            fontSize: 16, fontWeight: FontWeight.w600),
                      ),
                      Text(
                        _items.isNotEmpty ? "~${_items.first.daysLeft} days" : "",
                        style: const TextStyle(
                          color: primary,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 14),
                  Text(
                    _items.isNotEmpty ? _items.first.formatted : "No upcoming pickups",
                    style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 6),
                  const Text(
                    "Auto-scheduled: Every Monday & Thursday",
                    style: TextStyle(fontSize: 13),
                  ),
                  const SizedBox(height: 18),

                  /// REMIND BUTTON (NHỎ LẠI + CHỮ TRẮNG)
                  Center(
                    child: SizedBox(
                      width: 220,
                      height: 42,
                      child: ElevatedButton.icon(
                        style: ElevatedButton.styleFrom(
                          backgroundColor: primary,
                          foregroundColor: Colors.white,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(25),
                          ),
                          elevation: 0,
                        ),
                        icon: const Icon(
                          Icons.notifications,
                          size: 18,
                          color: Colors.white,
                        ),
                        onPressed: _items.isEmpty ? null : _toggleReminder,
                        label: Text(
                          _reminderEnabled ? "Reminder On" : "Remind Me",
                          style: const TextStyle(
                            fontWeight: FontWeight.w600,
                            color: Colors.white,
                          ),
                        ),
                      ),
                    ),
                  )
                ],
              ),
            ),

            const SizedBox(height: 26),

            const Text(
              "Upcoming Pickups",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700),
            ),

            const SizedBox(height: 14),

            for (final it in _items.skip(1).take(5))
              _pickupTile(it.formattedDate, 'Estimated · ${it.binId}', '${it.priority} · In ${it.daysLeft} days'),
          ],
        ),
        ),
      ),
    );
  }

  static Widget _pickupTile(String date, String subtitle, String badge) {
    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.06),
            blurRadius: 14,
            offset: const Offset(0, 8),
          )
        ],
      ),
      child: Row(
        children: [
          const Icon(Icons.access_time, color: primary, size: 28),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  date,
                  style: const TextStyle(
                      fontWeight: FontWeight.w600, fontSize: 15),
                ),
                const SizedBox(height: 4),
                Text(
                  subtitle,
                  style: const TextStyle(fontSize: 13),
                ),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
            decoration: BoxDecoration(
              color: const Color(0xFFE3F2E6),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Text(
              badge,
              style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600),
            ),
          )
        ],
      ),
    );
  }
}

class _PickupItem {
  final String binId;
  final int avgFill;
  final DateTime eta;
  final String priority;

  _PickupItem({required this.binId, required this.avgFill, required this.eta, required this.priority});

  int get daysLeft {
    final diff = eta.difference(DateTime.now()).inDays;
    return diff < 0 ? 0 : diff;
  }

  String get formatted {
    final h = eta.hour.toString().padLeft(2, '0');
    final m = eta.minute.toString().padLeft(2, '0');
    return '${eta.day}/${eta.month}/${eta.year} | $h:$m';
  }

  String get formattedDate => '${eta.day}/${eta.month}/${eta.year}';
}

import 'package:flutter/material.dart';
import '../utils/top_toast.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import '../services/api_service.dart';
import '../services/auth_service.dart';
import '../services/notification_service.dart';

class ScheduleScreen extends StatefulWidget {
  const ScheduleScreen({super.key});

  @override
  State<ScheduleScreen> createState() => _ScheduleScreenState();
}

class _ScheduleScreenState extends State<ScheduleScreen> {
  static const _reminderEnabledKey = 'pickup_reminder_enabled';
  static const _reminderAtKey = 'pickup_reminder_at';
  static const _reminderLeadKey = 'pickup_reminder_lead_minutes';

  final _authService = AuthService();
  final _storage = const FlutterSecureStorage();
  late final ApiService _api;
  bool _loading = true;
  bool _triggering = false;
  bool _reminderEnabled = false;
  int _selectedReminderLeadMinutes = 60;
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
        final predictedMs = _parseEpochMillis(row['predictedPickupAt']);
        final now = DateTime.now();
        final eta = predictedMs > 0
            ? _fromEpochMillisToLocal(predictedMs)
            : now.add(const Duration(days: 1));
        final priority = (row['priority'] ?? 'LOW').toString();
        return _PickupItem(binId: id, avgFill: avg, eta: eta, priority: priority);
      }).toList();

      if (!mounted) return;
      setState(() => _items = items);
      await _syncReminderStateWithSchedule(items);
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
      TopToast.show(context, 'Triggered aggregate. Refreshing schedule...');
      await Future<void>.delayed(const Duration(seconds: 2));
      await _load();
    } catch (_) {
      if (!mounted) return;
      TopToast.show(context, 'Cannot trigger aggregate right now.');
    } finally {
      if (mounted) setState(() => _triggering = false);
    }
  }

  DateTime _buildReminderTime(_PickupItem next, int leadMinutes) {
    final beforePickup = next.eta.subtract(Duration(minutes: leadMinutes));
    final minLead = DateTime.now().add(const Duration(seconds: 10));
    return beforePickup.isAfter(minLead) ? beforePickup : minLead;
  }

  String _leadLabel(int minutes) {
    switch (minutes) {
      case 10:
        return '10 minutes';
      case 30:
        return '30 minutes';
      case 60:
        return '1 hour';
      case 1440:
        return '1 day';
      default:
        return '$minutes minutes';
    }
  }

  Future<int?> _pickReminderLeadMinutes() async {
    int temp = _selectedReminderLeadMinutes;
    final picked = await showModalBottomSheet<int>(
      context: context,
      showDragHandle: true,
      builder: (ctx) {
        return StatefulBuilder(
          builder: (context, setLocalState) {
            return SafeArea(
              child: Padding(
                padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Remind me before pickup',
                      style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
                    ),
                    const SizedBox(height: 8),
                    RadioListTile<int>(
                      activeColor: primary,
                      value: 10,
                      groupValue: temp,
                      onChanged: (v) => setLocalState(() => temp = v ?? 10),
                      title: const Text('10 minutes'),
                    ),
                    RadioListTile<int>(
                      activeColor: primary,
                      value: 30,
                      groupValue: temp,
                      onChanged: (v) => setLocalState(() => temp = v ?? 30),
                      title: const Text('30 minutes'),
                    ),
                    RadioListTile<int>(
                      activeColor: primary,
                      value: 60,
                      groupValue: temp,
                      onChanged: (v) => setLocalState(() => temp = v ?? 60),
                      title: const Text('1 hour'),
                    ),
                    RadioListTile<int>(
                      activeColor: primary,
                      value: 1440,
                      groupValue: temp,
                      onChanged: (v) => setLocalState(() => temp = v ?? 1440),
                      title: const Text('1 day'),
                    ),
                    const SizedBox(height: 8),
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton(
                        style: ElevatedButton.styleFrom(
                          backgroundColor: primary, //  nền xanh
                          foregroundColor: Colors.white,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                        ),
                        onPressed: () => Navigator.of(ctx).pop(temp),
                        child: const Text('Confirm'),
                      ),
                    ),
                  ],
                ),
              ),
            );
          },
        );
      },
    );

    return picked;
  }

  Future<void> _syncReminderStateWithSchedule(List<_PickupItem> items) async {
    final enabled = await _storage.read(key: _reminderEnabledKey) == 'true';
    final reminderAtRaw = await _storage.read(key: _reminderAtKey);
    final leadRaw = await _storage.read(key: _reminderLeadKey);
    final reminderAtMs = int.tryParse(reminderAtRaw ?? '');
    final leadMinutes = int.tryParse(leadRaw ?? '') ?? _selectedReminderLeadMinutes;

    if (mounted && _selectedReminderLeadMinutes != leadMinutes) {
      setState(() => _selectedReminderLeadMinutes = leadMinutes);
    }

    if (!enabled || reminderAtMs == null || items.isEmpty) {
      if (!mounted) return;
      if (_reminderEnabled) {
        setState(() => _reminderEnabled = false);
      }
      return;
    }

    final expectedReminderAt =
        _buildReminderTime(items.first, leadMinutes).millisecondsSinceEpoch;
    final isSameSchedule = (reminderAtMs - expectedReminderAt).abs() < 60000;
    final isFuture = _fromEpochMillisToLocal(reminderAtMs).isAfter(DateTime.now());
    final stillEnabled = isSameSchedule && isFuture;

    if (stillEnabled) {
      if (!mounted) return;
      setState(() => _reminderEnabled = true);
      return;
    }

    final next = items.first;
    final reminderTime = _buildReminderTime(next, leadMinutes);

    try {
      await NotificationService.instance.schedulePickupReminder(
        scheduledAt: reminderTime,
        title: 'Trash pickup reminder',
        body: 'Bin ${next.binId} pickup at ${next.formatted}.',
      );
      await _storage.write(key: _reminderEnabledKey, value: 'true');
      await _storage.write(
        key: _reminderAtKey,
        value: reminderTime.millisecondsSinceEpoch.toString(),
      );

      if (!mounted) return;
      setState(() {
        _reminderEnabled = true;
        _selectedReminderLeadMinutes = leadMinutes;
      });
    } catch (_) {
      await NotificationService.instance.cancelPickupReminder();
      await _storage.write(key: _reminderEnabledKey, value: 'false');
      await _storage.delete(key: _reminderAtKey);
      await _storage.delete(key: _reminderLeadKey);
      if (!mounted) return;
      setState(() => _reminderEnabled = false);
    }
  }

  Future<void> _toggleReminder() async {
    if (_items.isEmpty) return;
    final next = _items.first;

    if (_reminderEnabled) {
      await NotificationService.instance.cancelPickupReminder();
      await _storage.write(key: _reminderEnabledKey, value: 'false');
      await _storage.delete(key: _reminderAtKey);
      await _storage.delete(key: _reminderLeadKey);
      if (!mounted) return;
      setState(() => _reminderEnabled = false);
      TopToast.show(context, 'Reminder disabled.');
      return;
    }

    try {
      final pickedLead = await _pickReminderLeadMinutes();
      if (pickedLead == null) return;

      final reminderTime = _buildReminderTime(next, pickedLead);
      await NotificationService.instance.schedulePickupReminder(
        scheduledAt: reminderTime,
        title: 'Trash pickup reminder',
        body: 'Bin ${next.binId} pickup at ${next.formatted}.',
      );
      await _storage.write(key: _reminderEnabledKey, value: 'true');
      await _storage.write(
        key: _reminderAtKey,
        value: reminderTime.millisecondsSinceEpoch.toString(),
      );
      await _storage.write(
        key: _reminderLeadKey,
        value: pickedLead.toString(),
      );

      if (!mounted) return;
      setState(() {
        _reminderEnabled = true;
        _selectedReminderLeadMinutes = pickedLead;
      });
      TopToast.show(context, 'Reminder set ${_leadLabel(pickedLead)}');
    } catch (_) {
      if (!mounted) return;
      TopToast.show(context, 'Cannot enable reminder right now. Please try again.');
    }
  }

  static int? _toInt(dynamic value) {
    if (value is int) return value;
    if (value is num) return value.toInt();
    return int.tryParse(value?.toString() ?? '');
  }

  static int _parseEpochMillis(dynamic value) {
    final parsed = _toInt(value) ?? 0;
    // Backward-compatible: accept both second and millisecond epoch values.
    if (parsed > 0 && parsed < 1000000000000) {
      return parsed * 1000;
    }
    return parsed;
  }

  static DateTime _fromEpochMillisToLocal(int epochMillis) {
    return DateTime.fromMillisecondsSinceEpoch(epochMillis, isUtc: true)
        .toLocal();
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
          'Trash collection schedule',
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
                                'No collection schedule available',
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
                    color: Colors.black.withOpacity(0.08),
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

                      const Spacer(),

                      Text(
                        _items.isNotEmpty ? _items.first.timeLeftLabel : "",
                        style: TextStyle(
                          color: _items.isNotEmpty && _items.first.eta.isBefore(DateTime.now()) 
                              ? Colors.red 
                              : primary,
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
                    "AI-estimated from latest sensor data",
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
                        onPressed: _items.isEmpty ? null : () => _toggleReminder(),
                        label: Text(
                          _reminderEnabled
                              ? 'Reminder On (${_leadLabel(_selectedReminderLeadMinutes)})'
                              : 'Remind Me',
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
              _pickupTile(
                it.formattedDate, 
                'Estimated · ${it.binId}', 
                it.eta.isBefore(DateTime.now()) 
                    ? 'Overdue'
                    : '${it.priority} · In ${it.timeLeftLabel}',
                isOverdue: it.eta.isBefore(DateTime.now())
              ),
          ],
        ),
        ),
      ),
    );
  }

  static Widget _pickupTile(String date, String subtitle, String badge, {bool isOverdue = false}) {
    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.06),
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
              color: isOverdue ? const Color(0xFFFFEBEB) : const Color(0xFFE3F2E6),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Text(
              badge,
              style: TextStyle(
                fontSize: 12, 
                fontWeight: FontWeight.w600,
                color: isOverdue ? Colors.red : Colors.black87,
              ),
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
    final diffMinutes = eta.difference(DateTime.now()).inMinutes;
    if (diffMinutes <= 0) return 0;
    return (diffMinutes / (24 * 60)).ceil();
  }

  String get timeLeftLabel {
    final now = DateTime.now();
    if (eta.isBefore(now)) return 'Already overdue';
    
    final diffMinutes = eta.difference(now).inMinutes;
    if (diffMinutes < 60) return '$diffMinutes minutes';
    if (diffMinutes < 1440) return '${diffMinutes ~/ 60} hours';
    final days = (diffMinutes / 1440).ceil();
    return '$days days';
  }

  String get formatted {
    final h = eta.hour.toString().padLeft(2, '0');
    final m = eta.minute.toString().padLeft(2, '0');
    return '${eta.day}/${eta.month}/${eta.year} | $h:$m';
  }

  String get formattedDate => '${eta.day}/${eta.month}/${eta.year}';
}

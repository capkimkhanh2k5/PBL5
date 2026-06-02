import 'package:flutter/material.dart';
import 'package:dio/dio.dart';
import 'bin_detail_screen.dart';
import 'settings_screen.dart';
import '../services/api_service.dart';
import '../services/auth_service.dart';
import 'package:flutter_slidable/flutter_slidable.dart';
import 'update_bin_screen.dart';
import '../utils/top_toast.dart';
import 'recycle_statistics_screen.dart';

class HomeScreen extends StatefulWidget {
  final ApiService apiService;

  const HomeScreen({
    super.key,
    required this.apiService,
  });

  @override
  State<HomeScreen> createState() => HomeScreenState();
}

class HomeScreenState extends State<HomeScreen> {

  final TextEditingController _searchCtrl = TextEditingController();
  final _authService = AuthService();
  String _query = '';
  bool _isLoading = true;
  String? _error;

  List<TrashCanItem> _items = const [];

  final Set<String> _commandLoadingBinIds = <String>{};

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  @override
  void dispose() {
    _searchCtrl.dispose();
    super.dispose();
  }

  Future<void> reloadBins() async {
    await _loadData();
  }

  Future<void> _loadData() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final api = widget.apiService;
      final results = await Future.wait([
        api.getAllBins(),
        api.getAllBinStatuses(),
      ]);

      final bins = results[0];
      final statuses = results[1];
      final statusById = <String, Map<String, dynamic>>{
        for (final s in statuses)
          if ((s['id'] ?? '').toString().isNotEmpty) (s['id'] ?? '').toString(): s,
      };

      final items = <TrashCanItem>[];

      for (final b in bins) {
        final id = (b['id'] ?? '').toString();
        if (id.isEmpty) continue;

        final status = statusById[id];
        final percent = _calcPercent(status);
        final lastUpdated = _toInt(status?['lastUpdated']);

        items.add(
          TrashCanItem(
            id: id,
            percent: percent,
            lastEmptiedText: _relativeTime(lastUpdated),
            classificationEnabled: _toBool(
              b['classificationEnabled'] ?? b['classification_enabled'],
            ) ??
                true,
          ),
        );
      }

      if (items.isEmpty && statuses.isNotEmpty) {
        for (final s in statuses) {
          final id = (s['id'] ?? '').toString();
          if (id.isEmpty) continue;
          items.add(
            TrashCanItem(
              id: id,
              percent: _calcPercent(s),
              lastEmptiedText: _relativeTime(_toInt(s['lastUpdated'])),
              classificationEnabled: true,
            ),
          );
        }
      }

      if (!mounted) return;
      setState(() => _items = items);
    } catch (e) {
      if (!mounted) return;
      if (e is DioException) {
        debugPrint(
          'Home load bins failed: type=${e.type}, status=${e.response?.statusCode}, '
          'uri=${e.requestOptions.uri}, message=${e.message}',
        );
      } else {
        debugPrint('Home load bins failed: $e');
      }
      setState(() => _error = 'Failed to load bins from backend. Please check API_BASE_URL and backend status.');
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  static int? _toInt(dynamic value) {
    if (value is int) return value;
    if (value is num) return value.toInt();
    return int.tryParse(value?.toString() ?? '');
  }

  static bool? _toBool(dynamic value) {
    if (value is bool) return value;
    if (value is num) return value != 0;

    final text = value?.toString().trim().toLowerCase();

    if (text == null || text.isEmpty) return null;

    if (['true', '1', 'on', 'yes'].contains(text)) return true;
    if (['false', '0', 'off', 'no'].contains(text)) return false;

    return null;
  }

  static double _calcPercent(Map<String, dynamic>? status) {
    if (status == null) return 0.0;
    final values = <int>[];
    for (final key in ['fillOrganic', 'fillRecycle', 'fillNonRecycle', 'fillHazardous']) {
      final v = _toInt(status[key]);
      if (v != null) values.add(v.clamp(0, 100));
    }
    if (values.isEmpty) return 0.0;
    final avg = values.reduce((a, b) => a + b) / values.length;
    return (avg / 100).clamp(0.0, 1.0);
  }

  static String _relativeTime(int? epochMillis) {
    if (epochMillis == null || epochMillis <= 0) {
      return 'Last updated: unknown';
    }
    final now = DateTime.now();
    final at = DateTime.fromMillisecondsSinceEpoch(epochMillis);
    final diff = now.difference(at);
    if (diff.inMinutes < 1) return 'Last updated: just now';
    if (diff.inHours < 1) return 'Last updated: ${diff.inMinutes}m ago';
    if (diff.inDays < 1) return 'Last updated: ${diff.inHours}h ago';
    return 'Last updated: ${diff.inDays}d ago';
  }

  @override
  Widget build(BuildContext context) {
    const bg = Colors.white;
    const headerH = 240.0;

    final currentUser = _authService.currentUser;
    final rawName = currentUser?.displayName ?? '';
    final rawEmail = currentUser?.email ?? '';
    
    final displayName = rawName.isNotEmpty 
        ? rawName 
        : (rawEmail.isNotEmpty ? rawEmail.split('@').first : 'User');

    final filtered = _items.where((e) {
      final q = _query.trim().toLowerCase();
      if (q.isEmpty) return true;
      return e.id.toLowerCase().contains(q);
    }).toList();

    return GestureDetector(
        behavior: HitTestBehavior.opaque, // 👈 QUAN TRỌNG
        onTap: () {
          FocusScope.of(context).unfocus();
        },
        child: Scaffold(
      backgroundColor: bg,
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 110), // chừa chỗ bottom bar
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header ảnh + greeting + search (giống mẫu plant)
              SizedBox(
                height: 502,
                child: Stack(
                  clipBehavior: Clip.none,
                  children: [
                    ClipRRect(
                      borderRadius: BorderRadius.circular(22),
                      child: SizedBox(
                        height: headerH,
                        width: double.infinity,
                        child: Stack(
                          fit: StackFit.expand,
                          children: [
                            Image.asset(
                              "assets/images/leaves.jpg",
                              fit: BoxFit.cover,
                            ),

                            Container(
                              color: Colors.black.withOpacity(0.18),
                            ),

                            Positioned(
                              left: 16,
                              top: 30,
                              right: 16,
                              child: Row(
                                children: [
                                  Expanded(
                                    child: Column(
                                      crossAxisAlignment: CrossAxisAlignment.start,
                                      children: [
                                        Text(
                                          "Hello, $displayName!",
                                          maxLines: 2,
                                          overflow: TextOverflow.ellipsis,
                                          style: const TextStyle(
                                            color: Colors.white,
                                            fontSize: 28,
                                            fontWeight: FontWeight.w800,
                                            height: 1.12,
                                          ),
                                        ),
                                        const SizedBox(height: 6),
                                        Row(
                                          children: [
                                            Icon(
                                              Icons.cloud_outlined,
                                              color: Colors.white.withOpacity(0.9),
                                              size: 18,
                                            ),
                                            const SizedBox(width: 6),
                                            Text(
                                              "Sun Cloudy 22°",
                                              style: TextStyle(
                                                color: Colors.white.withOpacity(0.9),
                                                fontSize: 15,
                                              ),
                                            ),
                                          ],
                                        ),
                                      ],
                                    ),
                                  ),

                                  const SizedBox(width: 12),

                                  InkWell(
                                    borderRadius: BorderRadius.circular(20),
                                    onTap: () {
                                      Navigator.push(
                                        context,
                                        MaterialPageRoute(
                                          builder: (_) => const SettingsScreen(),
                                        ),
                                      );
                                    },
                                    child: Container(
                                      padding: const EdgeInsets.all(8),
                                      decoration: BoxDecoration(
                                        color: Colors.white.withOpacity(0.2),
                                        borderRadius: BorderRadius.circular(20),
                                        border: Border.all(
                                          color: Colors.white.withOpacity(0.4),
                                        ),
                                      ),
                                      child: const Icon(
                                        Icons.settings,
                                        color: Colors.white,
                                        size: 18,
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                            ),

                            Positioned(
                              left: 16,
                              right: 16,
                              bottom: 42,
                              child: _SearchTextField(
                                controller: _searchCtrl,
                                hint: "Search bin, location...",
                                onChanged: (v) => setState(() => _query = v),
                                onClear: () {
                                  _searchCtrl.clear();
                                  setState(() => _query = '');
                                },
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),

                    Positioned(
                      left: 10,
                      right: 10,
                      top: headerH - 36,
                      child: RecycleWeeklySummaryCard(
                        apiService: widget.apiService,
                        onTap: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (_) => RecycleStatisticsScreen(
                                apiService: widget.apiService,
                              ),
                            ),
                          );
                        },
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 18),
              Row(
                children: [
                  const Expanded(
                    child: Text(
                      "All Smart Bins",
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.w900,
                      ),
                    ),
                  ),
                  Row(
                    children: [
                      const Icon(
                        Icons.filter_list,
                        size: 18,
                        color: Colors.black54,
                      ),
                      const SizedBox(width: 4),
                      Text(
                        "All",
                        style: TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w700,
                          color: Colors.black.withOpacity(0.65),
                        ),
                      ),
                      const Icon(
                        Icons.keyboard_arrow_down,
                        size: 18,
                        color: Colors.black54,
                      ),
                    ],
                  ),
                ],
              ),

              const SizedBox(height: 10),

              // List Trash can cards
              if (_isLoading)
                const Padding(
                  padding: EdgeInsets.symmetric(vertical: 30),
                  child: Center(child: CircularProgressIndicator()),
                )
              else if (_error != null)
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 20),
                  child: Column(
                    children: [
                      Text(_error!, style: const TextStyle(color: Colors.red)),
                      const SizedBox(height: 8),
                      ElevatedButton(
                        onPressed: _loadData,
                        child: const Text('Retry'),
                      ),
                    ],
                  ),
                )
              else if (filtered.isEmpty)
                const Padding(
                  padding: EdgeInsets.symmetric(vertical: 20),
                  child: Text('No bins found.'),
                )
              else
                ListView.separated(
                  itemCount: filtered.length,
                  shrinkWrap: true,
                  physics: const NeverScrollableScrollPhysics(),
                  separatorBuilder: (_, __) => const SizedBox(height: 12),
                  itemBuilder: (context, index) {
                    final it = filtered[index];
                    return TrashCanCard(
                      item: it,
                      isToggling: _commandLoadingBinIds.contains(it.id),
                      onToggleClassification: () => _toggleClassification(it),
                      onUpdate: () => _updateBin(it),
                      onDelete: () => onDelete(it.id),
                    );
                  },
                ),
            ],
          ),
        ),
      ),
    )
    );
  }


  Future<void> _toggleClassification(TrashCanItem item) async {
    final nextEnabled = !item.classificationEnabled;

    setState(() {
      _commandLoadingBinIds.add(item.id);

      _items = _items
          .map(
            (e) => e.id == item.id
            ? e.copyWith(classificationEnabled: nextEnabled)
            : e,
      )
          .toList();
    });

    try {
      await widget.apiService.sendClassificationCommand(
        item.id,
        enabled: nextEnabled,
      );

      if (!mounted) return;

      TopToast.show(
        context,
        nextEnabled
            ? 'Classification command sent: ON.'
            : 'Classification command sent: OFF.',
        type: ToastType.success,
      );
    } catch (e) {
      if (!mounted) return;

      setState(() {
        _items = _items
            .map(
              (x) => x.id == item.id
              ? x.copyWith(
            classificationEnabled: item.classificationEnabled,
          )
              : x,
        )
            .toList();
      });

      TopToast.show(
        context,
        'Error sending classification command.',
        type: ToastType.error,
      );
    } finally {
      if (mounted) {
        setState(() {
          _commandLoadingBinIds.remove(item.id);
        });
      }
    }
  }

  void _updateBin(TrashCanItem item) async {
    final updated = await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => UpdateBinScreen(
          binId: item.id,
          apiService: widget.apiService,
        ),
      ),
    );

    if (updated == true) {
      await _loadData();

      if (!mounted) return;

      TopToast.show(
        context,
        'Bin updated successfully.',
        type: ToastType.success,
      );
    }
  }

  void onDelete(String binId) async {
    const darkGreen = Color(0xFF0B5D1E);

    bool? confirmDelete = await showDialog<bool>(
      context: context,
      barrierDismissible: true,
      builder: (BuildContext context) {
        return AlertDialog(
          backgroundColor: Colors.white,
          surfaceTintColor: Colors.transparent,

          // số càng lớn thì khung càng nhỏ ngang
          insetPadding: const EdgeInsets.symmetric(horizontal: 60),

          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(18),
          ),

          titlePadding: const EdgeInsets.fromLTRB(24, 24, 24, 0),
          contentPadding: const EdgeInsets.fromLTRB(24, 18, 24, 16),
          actionsPadding: const EdgeInsets.fromLTRB(12, 0, 24, 18),

          title: const Text(
            "Delete",
            style: TextStyle(
              fontSize: 28,
              fontWeight: FontWeight.normal,
              color: Colors.black,
            ),
          ),

          content: const Text(
            "Do you want to delete this bin?",
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.normal,
              color: Colors.black87,
            ),
          ),

          actions: [
            TextButton(
              onPressed: () {
                Navigator.of(context).pop(false);
              },
              style: TextButton.styleFrom(
                foregroundColor: Colors.black,
                textStyle: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w500,
                ),
              ),
              child: const Text("No"),
            ),

            TextButton(
              onPressed: () {
                Navigator.of(context).pop(true);
              },
              style: TextButton.styleFrom(
                foregroundColor: darkGreen,
                textStyle: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w500,
                ),
              ),
              child: const Text("Yes"),
            ),
          ],
        );
      },
    );

    if (confirmDelete == true) {
      try {
        await widget.apiService.deleteBin(binId);

        setState(() {
          _items.removeWhere((item) => item.id == binId);
        });

        TopToast.show(
          context,
          'Bin deleted successfully.',
          type: ToastType.success,
        );
      } catch (e) {
        TopToast.show(
          context,
          'Error deleting item.',
          type: ToastType.error,
        );
      }
    }
  }
}

class _SearchTextField extends StatelessWidget {
  const _SearchTextField({
    required this.controller,
    required this.hint,
    required this.onChanged,
    required this.onClear,
  });

  final TextEditingController controller;
  final String hint;
  final ValueChanged<String> onChanged;
  final VoidCallback onClear;

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(18),
      child: Container(
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.18),
          border: Border.all(color: Colors.white.withOpacity(0.25)),
        ),
        child: TextField(
          controller: controller,
          onChanged: onChanged,
          style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w700),
          cursorColor: Colors.white,
          decoration: InputDecoration(
            prefixIcon: Icon(Icons.search, color: Colors.white.withOpacity(0.9)),
            hintText: hint,
            hintStyle: TextStyle(color: Colors.white.withOpacity(0.75), fontWeight: FontWeight.w700),
            border: InputBorder.none,
            contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 14),
            suffixIcon: controller.text.isEmpty
                ? null
                : IconButton(
              onPressed: onClear,
              icon: Icon(Icons.close, color: Colors.white.withOpacity(0.9)),
            ),
          ),
        ),
      ),
    );
  }
}


class TrashCanItem {
  final String id;
  final double percent; // 0..1
  final String lastEmptiedText;
  final bool classificationEnabled;

  const TrashCanItem({
    required this.id,
    required this.percent,
    required this.lastEmptiedText,
    required this.classificationEnabled,
  });

  TrashCanItem copyWith({
    String? id,
    double? percent,
    String? lastEmptiedText,
    bool? classificationEnabled,
  }) {
    return TrashCanItem(
      id: id ?? this.id,
      percent: percent ?? this.percent,
      lastEmptiedText: lastEmptiedText ?? this.lastEmptiedText,
      classificationEnabled:
      classificationEnabled ?? this.classificationEnabled,
    );
  }
}

class TrashCanCard extends StatelessWidget {
  const TrashCanCard({
    super.key,
    required this.item,
    required this.isToggling,
    required this.onToggleClassification,
    required this.onUpdate,
    required this.onDelete,
  });

  final TrashCanItem item;
  final bool isToggling;
  final VoidCallback onToggleClassification;
  final VoidCallback onUpdate;
  final VoidCallback onDelete;

  @override
  Widget build(BuildContext context) {
    const green = Color(0xFF2F6B3D);
    const plantTile = Color(0xFFB9D98A);

    final pctText = "${(item.percent * 100).round()}%";

    return Slidable(
      key: ValueKey(item.id),

      endActionPane: ActionPane(
        motion: const DrawerMotion(),
        extentRatio: 0.45,
        children: [
          SlidableAction(
            onPressed: (_) => onUpdate(),
            backgroundColor: const Color(0xFFFFB74D),
            foregroundColor: Colors.white,
            icon: Icons.edit,
            borderRadius: BorderRadius.circular(18),
          ),
          SlidableAction(
            onPressed: (_) => onDelete(),
            backgroundColor: Colors.red,
            foregroundColor: Colors.white,
            icon: Icons.delete,
            borderRadius: BorderRadius.circular(18),
          ),
        ],
      ),

      child: InkWell(
        borderRadius: BorderRadius.circular(18),
        onTap: () {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (_) => BinDetailScreen(binId: item.id),
            ),
          );
        },
        child: Container(
          padding: const EdgeInsets.fromLTRB(14, 14, 14, 14),
          decoration: BoxDecoration(
            color: plantTile,
            borderRadius: BorderRadius.circular(18),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.06),
                blurRadius: 18,
                offset: const Offset(0, 10),
              ),
            ],
          ),
          child: Row(
            children: [
              const Icon(
                Icons.delete_outline,
                color: Colors.black87,
              ),

              const SizedBox(width: 10),

              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      item.id,
                      style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w900,
                      ),
                    ),

                    const SizedBox(height: 4),

                    Text(
                      item.lastEmptiedText,
                      style: TextStyle(
                        color: Colors.black.withOpacity(0.65),
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ),

              // Độ đầy nằm bên trái nút bật/tắt
              _RingPercent(
                percent: item.percent,
                text: pctText,
                color: green,
              ),

              const SizedBox(width: 8),

              // Nút ON/OFF nằm ngoài cùng bên phải
              _ClassificationToggle(
                enabled: item.classificationEnabled,
                isLoading: isToggling,
                onTap: onToggleClassification,
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ClassificationToggle extends StatelessWidget {
  const _ClassificationToggle({
    required this.enabled,
    required this.isLoading,
    required this.onTap,
  });

  final bool enabled;
  final bool isLoading;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    const green = Color(0xFF0B5D1E);
    const offGray = Color(0xFF8B8B8B);

    final bgColor = enabled ? green : offGray;
    final label = enabled ? 'ON' : 'OFF';

    return Material(
      color: Colors.transparent,
      child: InkWell(
        borderRadius: BorderRadius.circular(999),
        onTap: isLoading ? null : onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 180),
          curve: Curves.easeOut,
          width: 68,
          height: 30,
          padding: const EdgeInsets.all(4),
          decoration: BoxDecoration(
            color: bgColor,
            borderRadius: BorderRadius.circular(999),
            boxShadow: [
              BoxShadow(
                color: bgColor.withOpacity(0.18),
                blurRadius: 8,
                offset: const Offset(0, 4),
              ),
            ],
          ),
          child: Stack(
            alignment: Alignment.center,
            children: [
              Align(
                alignment:
                enabled ? Alignment.centerLeft : Alignment.centerRight,
                child: Padding(
                  padding: EdgeInsets.only(
                    left: enabled ? 8 : 0,
                    right: enabled ? 0 : 8,
                  ),
                  child: Text(
                    isLoading ? '...' : label,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 12,
                      fontWeight: FontWeight.w900,
                      letterSpacing: 0.6,
                    ),
                  ),
                ),
              ),

              AnimatedAlign(
                duration: const Duration(milliseconds: 180),
                curve: Curves.easeOut,
                alignment:
                enabled ? Alignment.centerRight : Alignment.centerLeft,
                child: Container(
                  width: 24,
                  height: 24,
                  decoration: BoxDecoration(
                    color: Colors.white,
                    shape: BoxShape.circle,
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.18),
                        blurRadius: 6,
                        offset: const Offset(0, 2),
                      ),
                    ],
                  ),
                  child: isLoading
                      ? const Padding(
                    padding: EdgeInsets.all(7),
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      color: green,
                    ),
                  )
                      : null,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _RingPercent extends StatelessWidget {
  const _RingPercent({
    required this.percent,
    required this.text,
    required this.color,
  });

  final double percent;
  final String text;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 54,
      height: 54,
      child: Stack(
        alignment: Alignment.center,
        children: [
          CircularProgressIndicator(
            value: percent.clamp(0, 1),
            strokeWidth: 6,
            backgroundColor: Colors.white.withOpacity(0.55),
            valueColor: AlwaysStoppedAnimation(color),
          ),
          Text(
            text,
            style: const TextStyle(fontSize: 10, fontWeight: FontWeight.w900),
          ),
        ],
      ),
    );
  }
}

class RecycleWeeklySummaryCard extends StatefulWidget {
  const RecycleWeeklySummaryCard({
    super.key,
    required this.apiService,
    required this.onTap,
  });

  final ApiService apiService;
  final VoidCallback onTap;

  @override
  State<RecycleWeeklySummaryCard> createState() =>
      _RecycleWeeklySummaryCardState();
}

class _RecycleWeeklySummaryCardState extends State<RecycleWeeklySummaryCard> {
  static const darkGreen = Color(0xFF0B5D1E);
  static const lightGreen = Color(0xFFEAF6C8);

  bool _isLoading = true;
  String? _error;

  List<_RecycleDay> _data = [];
  double _totalLiters = 0.0;
  int _percentChange = 0;

  @override
  void initState() {
    super.initState();
    _loadWeeklySummary();
  }

  Future<void> _loadWeeklySummary() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final now = DateTime.now();
      final today = DateTime(now.year, now.month, now.day);
      final start = today.subtract(const Duration(days: 6));

      final result = await widget.apiService.getWeeklyRecycleStatistics(
        startDate: _formatDate(start),
        endDate: _formatDate(today),
      );

      final rawDays = result['days'];

      final days = rawDays is List
          ? rawDays.map((item) {
        final map = Map<String, dynamic>.from(item as Map);

        return _RecycleDay(
          (map['label'] ?? '').toString(),
          _toDouble(map['liters']),
        );
      }).toList()
          : <_RecycleDay>[];

      if (!mounted) return;

      setState(() {
        _data = days;
        _totalLiters = _toDouble(result['totalLiters']);
        _percentChange = _toInt(result['percentChange']);
        _isLoading = false;
      });
    } catch (e) {
      if (!mounted) return;

      setState(() {
        _error = 'Failed to load recycle summary.';
        _isLoading = false;
      });
    }
  }

  String _formatDate(DateTime date) {
    final y = date.year.toString().padLeft(4, '0');
    final m = date.month.toString().padLeft(2, '0');
    final d = date.day.toString().padLeft(2, '0');

    return '$y-$m-$d';
  }

  static double _toDouble(dynamic value) {
    if (value == null) return 0.0;
    if (value is int) return value.toDouble();
    if (value is double) return value;
    if (value is num) return value.toDouble();
    return double.tryParse(value.toString()) ?? 0.0;
  }

  static int _toInt(dynamic value) {
    if (value == null) return 0;
    if (value is int) return value;
    if (value is double) return value.round();
    if (value is num) return value.round();
    return int.tryParse(value.toString()) ?? 0;
  }

  @override
  Widget build(BuildContext context) {
    final maxValue = _data.isEmpty
        ? 1.0
        : _data.map((e) => e.liters).reduce((a, b) => a > b ? a : b);

    final safeMaxValue = maxValue <= 0 ? 1.0 : maxValue;

    return InkWell(
      borderRadius: BorderRadius.circular(18),
      onTap: widget.onTap,
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.fromLTRB(14, 12, 14, 12),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(18),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.10),
              blurRadius: 24,
              spreadRadius: 1,
              offset: const Offset(0, 12),
            ),
          ],
        ),
        child: _isLoading
            ? const SizedBox(
          height: 245,
          child: Center(
            child: CircularProgressIndicator(
              color: darkGreen,
            ),
          ),
        )
            : _error != null
            ? SizedBox(
          height: 245,
          child: Center(
            child: Text(
              _error!,
              textAlign: TextAlign.center,
              style: const TextStyle(
                color: Colors.red,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
        )
            : Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Expanded(
                  child: Text(
                    'Weekly Recycle Summary',
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w900,
                      color: Colors.black87,
                    ),
                  ),
                ),
                const Text(
                  'View details',
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w800,
                    color: darkGreen,
                  ),
                ),
                const SizedBox(width: 3),
                const Icon(
                  Icons.arrow_forward_ios,
                  size: 9,
                  color: darkGreen,
                ),
              ],
            ),

            const SizedBox(height: 9),

            const Text(
              'Total recycled',
              style: TextStyle(
                fontSize: 11.5,
                color: Colors.black54,
                fontWeight: FontWeight.w600,
              ),
            ),

            const SizedBox(height: 2),

            Row(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Text(
                  _totalLiters.toStringAsFixed(1),
                  style: const TextStyle(
                    fontSize: 32,
                    fontWeight: FontWeight.w900,
                    color: Colors.black,
                    height: 1,
                  ),
                ),
                const SizedBox(width: 5),
                const Padding(
                  padding: EdgeInsets.only(bottom: 4),
                  child: Text(
                    'L',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                ),
              ],
            ),

            const SizedBox(height: 4),

            Row(
              children: [
                Icon(
                  _percentChange >= 0
                      ? Icons.arrow_upward
                      : Icons.arrow_downward,
                  size: 13,
                  color: darkGreen,
                ),
                const SizedBox(width: 3),
                Text(
                  '${_percentChange >= 0 ? '+' : ''}$_percentChange% vs last week',
                  style: const TextStyle(
                    fontSize: 11.5,
                    color: darkGreen,
                    fontWeight: FontWeight.w800,
                  ),
                ),
              ],
            ),

            const SizedBox(height: 10),

            SizedBox(
              height: 124,
              width: double.infinity,
              child: _data.isEmpty
                  ? const Center(
                child: Text(
                  'No recycle data this week.',
                  style: TextStyle(
                    color: Colors.black54,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              )
                  : Center(
                child: FractionallySizedBox(
                  widthFactor: 0.82,
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.end,
                    children: _data.map((item) {
                      final barHeight =
                          72 * (item.liters / safeMaxValue);

                      return Expanded(
                        child: Column(
                          mainAxisAlignment:
                          MainAxisAlignment.end,
                          children: [
                            Text(
                              item.liters.toStringAsFixed(1),
                              style: const TextStyle(
                                fontSize: 10.5,
                                fontWeight: FontWeight.w800,
                                color: Colors.black87,
                              ),
                            ),
                            const SizedBox(height: 6),
                            _FancyBar(height: barHeight),
                            const SizedBox(height: 8),
                            Text(
                              item.label,
                              style: TextStyle(
                                fontSize: 11,
                                fontWeight: FontWeight.w700,
                                color: Colors.black
                                    .withOpacity(0.65),
                              ),
                            ),
                          ],
                        ),
                      );
                    }).toList(),
                  ),
                ),
              ),
            ),

            const SizedBox(height: 9),

            Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(
                horizontal: 10,
                vertical: 7,
              ),
              decoration: BoxDecoration(
                color: lightGreen.withOpacity(0.75),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Row(
                children: [
                  const Icon(
                    Icons.eco,
                    size: 15,
                    color: darkGreen,
                  ),
                  const SizedBox(width: 6),
                  Expanded(
                    child: Text(
                      _percentChange >= 0
                          ? 'Great job! You recycled more than last week.'
                          : 'Recycled waste decreased compared with last week. Keep improving!',
                      style: const TextStyle(
                        fontSize: 11.5,
                        fontWeight: FontWeight.w800,
                        color: darkGreen,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _RecycleDay {
  final String label;
  final double liters;

  const _RecycleDay(this.label, this.liters);
}

class _FancyBar extends StatelessWidget {
  const _FancyBar({
    required this.height,
  });

  final double height;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 18,
      height: height,
      decoration: BoxDecoration(
        // Không bo góc để cột vuông giống ảnh mẫu
        borderRadius: BorderRadius.zero,
        gradient: const LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [
            Color(0xFF8BC34A),
            Color(0xFF65A936),
            Color(0xFF3F8F2F),
          ],
        ),
        boxShadow: [
          BoxShadow(
            color: const Color(0xFF3F8F2F).withOpacity(0.18),
            blurRadius: 5,
            offset: const Offset(0, 3),
          ),
        ],
      ),
    );
  }
}
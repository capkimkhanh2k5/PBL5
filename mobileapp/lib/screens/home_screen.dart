import 'package:flutter/material.dart';
import 'package:dio/dio.dart';
import 'bin_detail_screen.dart';
import 'ai_chat_screen.dart';
import 'settings_screen.dart';
import '../services/api_service.dart';
import '../services/auth_service.dart';
import 'package:flutter_slidable/flutter_slidable.dart';
import 'update_bin_screen.dart';
import '../utils/top_toast.dart';

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
    const headerH = 210.0;

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
                      // overlay nhẹ để chữ nổi
                      Container(color: Colors.black.withOpacity(0.18)),

                      Positioned(
                        left: 16,
                        top: 30,
                        right: 16,
                        child: Row(
                          children: [
                            // LEFT: Greeting
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    "Hello, $displayName!",
                                    style: const TextStyle(
                                      color: Colors.white,
                                      fontSize: 30,
                                      fontWeight: FontWeight.w800,
                                    ),
                                  ),
                                  const SizedBox(height: 6),
                                  Row(
                                    children: [
                                      Icon(Icons.cloud_outlined,
                                          color: Colors.white.withOpacity(0.9), size: 18),
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

                            // RIGHT: SETTINGS + LOGOUT
                            Row(
                              children: [
                                // SETTINGS
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
                                      border: Border.all(color: Colors.white.withOpacity(0.4)),
                                    ),
                                    child: const Icon(Icons.settings,
                                        color: Colors.white, size: 18),
                                  ),
                                ),

                                const SizedBox(width: 8),
                              ],
                            ),
                          ],
                        ),
                      ),


                      // Search bar (gõ được)
                      Positioned(
                        left: 16,
                        right: 16,
                        bottom: 16,
                        child: _SearchTextField(
                          controller: _searchCtrl,
                          hint: "Search",
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

              const SizedBox(height: 14),

              // Ask your plant card
              AskAiCard(apiService: widget.apiService),

              const SizedBox(height: 16),

              const Text(
                "All",
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.w900),
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

class AskAiCard extends StatelessWidget {
  final ApiService apiService;

  const AskAiCard({
    super.key,
    required this.apiService,
  });

  @override
  Widget build(BuildContext context) {
    const bg = Color(0xFFEAF6C8);

    return InkWell(
      borderRadius: BorderRadius.circular(18),
      onTap: () {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (_) => AiChatScreen(
              apiService: apiService,
            ),
          ),
        );
      },
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: bg,
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
            const Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    "Ask SmartBin\nAI Assistant!",
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w900,
                      height: 1.05,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    "Check fill levels, schedules, history\nand map locations instantly.",
                    style: TextStyle(fontSize: 12.5, height: 1.25),
                  ),
                ],
              ),
            ),
            const SizedBox(width: 12),
            Container(
              width: 70,
              height: 70,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: const LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [
                    Color(0xFFFFE8F7D6),
                    Color(0xFFFFD4EDB2),
                  ],
                ),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.10),
                    blurRadius: 12,
                    offset: const Offset(0, 8),
                  ),
                ],
              ),
              child: const Icon(Icons.smart_toy_outlined, size: 30),
            ),
          ],
        ),
      ),
    );
  }
}

class TrashCanItem {
  final String id;
  final double percent; // 0..1
  final String lastEmptiedText;

  const TrashCanItem({
    required this.id,
    required this.percent,
    required this.lastEmptiedText,
  });
}

class TrashCanCard extends StatelessWidget {
  const TrashCanCard({
    super.key,
    required this.item,
    required this.onUpdate,
    required this.onDelete,
  });

  final TrashCanItem item;
  final VoidCallback onUpdate;
  final VoidCallback onDelete;

  @override
  Widget build(BuildContext context) {
    const green = Color(0xFF2F6B3D);
    const plantTile = Color(0xFFB9D98A);

    final pctText = "${(item.percent * 100).round()}%";

    return Slidable(
      key: ValueKey(item.id),

      // Vuốt item sang trái sẽ hiện nút bên phải
      endActionPane: ActionPane(
        motion: const DrawerMotion(),
        extentRatio: 0.45,
        children: [
          SlidableAction(
            onPressed: (_) => onUpdate(),
            backgroundColor: Color(0xFFFFB74D),
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
          )
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
              const Icon(Icons.delete_outline, color: Colors.black87),
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

              _RingPercent(
                percent: item.percent,
                text: pctText,
                color: green,
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
            style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w900),
          ),
        ],
      ),
    );
  }
}
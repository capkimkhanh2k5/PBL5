import 'package:flutter/material.dart';
import '../services/api_service.dart';
import '../services/auth_service.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key, required this.binId});
  final String binId;

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  final _authService = AuthService();

  bool _loading = true;
  String? _error;
  List<TrashHistoryItem> _items = const [];
  String _selectedType = 'ALL';

  final List<Map<String, String>> _types = [
    {'key': 'ALL', 'label': 'Tất Cả'},
    {'key': 'General_Waste', 'label': 'Rác Chung'},
    {'key': 'ORGANIC', 'label': 'Hữu Cơ'},
    {'key': 'RECYCLABLE', 'label': 'Tái Chế'},
    {'key': 'HAZARDOUS', 'label': 'Nguy Hiểm'},
  ];

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final api = ApiService(authService: _authService);

      final Map<String, List<String>> categoryTypes = {
        'General_Waste': ['General_Waste', 'GENERAL'],
        'ORGANIC': ['ORGANIC', 'BIOLOGICAL'],
        'RECYCLABLE': [
          'RECYCLABLE',
          'PLASTIC',
          'Plastic',
          'Paper_Cardboard',
          'PAPER_CARDBOARD',
          'METAL',
          'GLASS',
        ],
        'HAZARDOUS': ['HAZARDOUS', 'Battery', 'BATTERY'],
      };

      List<Map<String, dynamic>> logs = [];

      if (_selectedType == 'ALL') {
        for (var entry in categoryTypes.entries) {
          for (var type in entry.value) {
            final typeLogs = await api.getClassificationLogs(
              binId: widget.binId,
              type: type,
              limit: 10,
            );

            logs.addAll(typeLogs);
          }
        }
      } else {
        final types = categoryTypes[_selectedType] ?? [_selectedType];

        for (var type in types) {
          final typeLogs = await api.getClassificationLogs(
            binId: widget.binId,
            type: type,
            limit: 10,
          );

          logs.addAll(typeLogs);
        }
      }

      final uniqueLogs = <String, Map<String, dynamic>>{};

      for (var log in logs) {
        uniqueLogs[(log['log_id'] ?? log['logId']).toString()] = log;
      }

      logs = uniqueLogs.values.toList();

      logs.sort((a, b) {
        final aTime =
            _toDateTime(a['classified_at'] ?? a['classifiedAt'])
                ?.millisecondsSinceEpoch ??
                0;

        final bTime =
            _toDateTime(b['classified_at'] ?? b['classifiedAt'])
                ?.millisecondsSinceEpoch ??
                0;

        return bTime.compareTo(aTime);
      });

      logs = logs.take(10).toList();

      final items = logs.map((e) {
        final rawTitle =
        ((e['classification_result'] ?? e['classificationResult']) ??
            'Unknown')
            .toString();

        final upperTitle = rawTitle.trim().toUpperCase().replaceAll(' ', '_');

        final labelMap = {
          'GENERAL_WASTE': 'Rác chung',
          'GENERAL': 'Rác chung',

          'BIOLOGICAL': 'Hữu cơ',
          'ORGANIC': 'Hữu cơ',

          'PLASTIC': 'Plastic',
          'PAPER_CARDBOARD': 'Paper/Cardboard',
          'PAPER': 'Paper',
          'CARDBOARD': 'Cardboard',
          'RECYCLABLE': 'Recyclable',
          'METAL': 'Metal',
          'GLASS': 'Glass',

          'BATTERY': 'Battery',
          'HAZARDOUS': 'Hazardous',
        };

        print('FULL ITEM: $e');

        return TrashHistoryItem(
          imageUrl: ((e['image_url'] ?? e['imageUrl']) ?? '').toString(),
          title: labelMap[upperTitle] ?? rawTitle,
          confidence: _toDouble(e['confidence_score'] ?? e['confidenceScore']),
          classifiedAt: _toDateTime(e['classified_at'] ?? e['classifiedAt']),
        );
      }).toList();

      if (!mounted) return;
      setState(() => _items = items);
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = 'Failed to load history from backend.');
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  static double? _toDouble(dynamic value) {
    if (value is double) return value;
    if (value is num) return value.toDouble();
    return double.tryParse(value?.toString() ?? '');
  }

  static DateTime? _toDateTime(dynamic value) {
    if (value == null) return null;

    if (value is Timestamp) {
      return value.toDate();
    }

    if (value is Map<Object?, Object?>) {
      final seconds = value['seconds'] ?? value['_seconds'];

      if (seconds != null) {
        return DateTime.fromMillisecondsSinceEpoch(
          (seconds as num).toInt() * 1000,
        );
      }
    }

    if (value is String) {
      return DateTime.tryParse(value);
    }

    return null;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFEAF6EE),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1B5E20),
        elevation: 0,
        foregroundColor: Colors.white,
        centerTitle: true,
        title: Text(
          'Trash history - ${widget.binId}',
          style: const TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.w800,
            fontSize: 18,
          ),
        ),
      ),
      body: Column(
        children: [
          Container(
            color: Colors.white,
            height: 64,
            padding: const EdgeInsets.symmetric(vertical: 10),
            child: ListView.separated(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              scrollDirection: Axis.horizontal,
              itemCount: _types.length,
              separatorBuilder: (_, __) => const SizedBox(width: 10),
              itemBuilder: (context, index) {
                final type = _types[index];
                final isSelected = _selectedType == type['key'];

                return ChoiceChip(
                  label: Text(type['label']!),
                  selected: isSelected,
                  showCheckmark: false,
                  backgroundColor: Colors.white,
                  selectedColor: const Color(0xFF2E7D32),
                  side: BorderSide(
                    color: isSelected
                        ? const Color(0xFF2E7D32)
                        : Colors.grey.shade300,
                  ),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(999),
                  ),
                  labelStyle: TextStyle(
                    color: isSelected ? Colors.white : Colors.black87,
                    fontWeight: FontWeight.w700,
                  ),
                  onSelected: (selected) {
                    if (selected && _selectedType != type['key']) {
                      setState(() => _selectedType = type['key']!);
                      _load();
                    }
                  },
                );
              },
            ),
          ),
          Expanded(
            child: _loading
                ? const Center(
              child: CircularProgressIndicator(
                color: Color(0xFF2E7D32),
              ),
            )
                : _error != null
                ? Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(
                    Icons.error_outline,
                    color: Colors.red,
                    size: 42,
                  ),
                  const SizedBox(height: 10),
                  Text(
                    _error!,
                    style: const TextStyle(
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: 12),
                  ElevatedButton(
                    onPressed: _load,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF2E7D32),
                      foregroundColor: Colors.white,
                    ),
                    child: const Text('Retry'),
                  ),
                ],
              ),
            )
                : _items.isEmpty
                ? const Center(
              child: Text(
                'No classification images yet.',
                style: TextStyle(
                  color: Colors.black54,
                  fontWeight: FontWeight.w600,
                ),
              ),
            )
                : ListView.separated(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
              itemCount: _items.length,
              separatorBuilder: (_, __) =>
              const SizedBox(height: 10),
              itemBuilder: (context, i) =>
                  _HistoryCard(item: _items[i]),
            ),
          ),
        ],
      ),
    );
  }
}

class TrashHistoryItem {
  final String imageUrl;
  final String title;
  final double? confidence;
  final DateTime? classifiedAt;

  const TrashHistoryItem({
    required this.imageUrl,
    required this.title,
    this.confidence,
    this.classifiedAt,
  });
}

class _HistoryCard extends StatelessWidget {
  const _HistoryCard({required this.item});
  final TrashHistoryItem item;

  IconData get _icon {
    switch (item.title) {
      case 'Rác chung':
        return Icons.delete;

      case 'Hữu cơ':
        return Icons.eco;

      case 'Plastic':
      case 'Paper/Cardboard':
      case 'Paper':
      case 'Cardboard':
      case 'Recyclable':
      case 'Metal':
      case 'Glass':
        return Icons.recycling;

      case 'Battery':
      case 'Hazardous':
        return Icons.warning_rounded;

      default:
        return Icons.help_outline;
    }
  }

  Color get _color {
    switch (item.title) {
      case 'Rác chung':
        return const Color(0xFF2E7D32);

      case 'Hữu cơ':
        return const Color(0xFF43A047);

      case 'Plastic':
      case 'Paper/Cardboard':
      case 'Paper':
      case 'Cardboard':
      case 'Recyclable':
      case 'Metal':
      case 'Glass':
        return const Color(0xFF1E88E5);

      case 'Battery':
      case 'Hazardous':
        return const Color(0xFFE53935);

      default:
        return Colors.grey;
    }
  }

  String get _confidenceText {
    if (item.confidence == null) return 'N/A';
    return '${(item.confidence! * 100).toStringAsFixed(2)}%';
  }

  String get _dateText {
    final date = item.classifiedAt;
    if (date == null) return '--/--/----';

    final day = date.day.toString().padLeft(2, '0');
    final month = date.month.toString().padLeft(2, '0');
    final year = date.year.toString();

    return '$day/$month/$year';
  }

  String get _timeText {
    final date = item.classifiedAt;
    if (date == null) return '--:--';

    final hour = date.hour.toString().padLeft(2, '0');
    final minute = date.minute.toString().padLeft(2, '0');

    return '$hour:$minute';
  }

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.white,
      borderRadius: BorderRadius.circular(16),
      elevation: 1.5,
      shadowColor: Colors.black.withOpacity(0.08),
      child: InkWell(
        borderRadius: BorderRadius.circular(16),
        onTap: () {},
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 9),
          child: Row(
            children: [
              ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: Image(
                  image: item.imageUrl.isNotEmpty
                      ? NetworkImage(item.imageUrl)
                      : const AssetImage('assets/images/leaves.jpg')
                  as ImageProvider,
                  width: 62,
                  height: 62,
                  fit: BoxFit.cover,
                  errorBuilder: (_, __, ___) => Image.asset(
                    'assets/images/leaves.jpg',
                    width: 62,
                    height: 62,
                    fit: BoxFit.cover,
                  ),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(
                          _icon,
                          color: _color,
                          size: 19,
                        ),
                        const SizedBox(width: 6),
                        Expanded(
                          child: Text(
                            item.title,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: TextStyle(
                              color: _color,
                              fontSize: 15.5,
                              fontWeight: FontWeight.w800,
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 5),
                    RichText(
                      text: TextSpan(
                        style: const TextStyle(
                          color: Color(0xFF5F6B7A),
                          fontSize: 13,
                          fontWeight: FontWeight.w500,
                        ),
                        children: [
                          const TextSpan(text: 'Confidence: '),
                          TextSpan(
                            text: _confidenceText,
                            style: TextStyle(
                              color: _color,
                              fontWeight: FontWeight.w800,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text(
                    _dateText,
                    style: const TextStyle(
                      color: Color(0xFF5F6B7A),
                      fontSize: 12.5,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: 5),
                  Text(
                    _timeText,
                    style: const TextStyle(
                      color: Color(0xFF5F6B7A),
                      fontSize: 12.5,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
              const SizedBox(width: 6),
              const Icon(
                Icons.chevron_right,
                color: Colors.grey,
                size: 24,
              ),
            ],
          ),
        ),
      ),
    );
  }
}
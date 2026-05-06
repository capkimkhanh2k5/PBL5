import 'package:flutter/material.dart';
import '../services/api_service.dart';
import '../services/auth_service.dart';

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
  String _selectedType = 'GENERAL';

  final List<Map<String, String>> _types = [
    {'key': 'GENERAL', 'label': 'Rác Chung'},
    {'key': 'BIOLOGICAL', 'label': 'Hữu Cơ'},
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
      final logs = await api.getClassificationLogs(
        binId: widget.binId,
        type: _selectedType,
        limit: 10,
      );
      final items = logs.map((e) {
        final rawTitle = ((e['classificationResult'] ?? e['classification_result']) ?? 'Unknown').toString();
        // Map raw title to label for display
        final labelMap = {
          'GENERAL': 'Rác Chung',
          'BIOLOGICAL': 'Hữu Cơ',
          'RECYCLABLE': 'Tái Chế',
          'HAZARDOUS': 'Nguy Hiểm',
        };
        return TrashHistoryItem(
          imageUrl: ((e['imageUrl'] ?? e['image_url']) ?? '').toString(),
          title: labelMap[rawTitle.toUpperCase()] ?? rawTitle,
          confidence: _toDouble(e['confidenceScore'] ?? e['confidence_score']),
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFEAF6EE),
      appBar: AppBar(
        backgroundColor: const Color(0xFF2E7D32),
        elevation: 0,
        foregroundColor: Colors.white,
        title: Text(
          'Trash history - ${widget.binId}',
          style: const TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.w700,
          ),
        ),
      ),
      body: Column(
        children: [
          Container(
            height: 60,
            padding: const EdgeInsets.symmetric(vertical: 8),
            child: ListView.separated(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              scrollDirection: Axis.horizontal,
              itemCount: _types.length,
              separatorBuilder: (_, __) => const SizedBox(width: 8),
              itemBuilder: (context, index) {
                final type = _types[index];
                final isSelected = _selectedType == type['key'];
                return ChoiceChip(
                  label: Text(type['label']!),
                  selected: isSelected,
                  selectedColor: const Color(0xFF2E7D32),
                  labelStyle: TextStyle(
                    color: isSelected ? Colors.white : Colors.black87,
                    fontWeight: FontWeight.w600,
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
                ? const Center(child: CircularProgressIndicator())
                : _error != null
                    ? Center(
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Text(_error!),
                            const SizedBox(height: 10),
                            ElevatedButton(onPressed: _load, child: const Text('Retry')),
                          ],
                        ),
                      )
                    : _items.isEmpty
                        ? const Center(child: Text('No classification images yet.'))
                        : ListView.separated(
                            padding: const EdgeInsets.fromLTRB(16, 0, 16, 24),
                            itemCount: _items.length,
                            separatorBuilder: (_, __) => const SizedBox(height: 18),
                            itemBuilder: (context, i) => _HistoryCard(item: _items[i]),
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

  const TrashHistoryItem({required this.imageUrl, required this.title, this.confidence});
}

class _HistoryCard extends StatelessWidget {
  const _HistoryCard({required this.item});
  final TrashHistoryItem item;

  @override
  Widget build(BuildContext context) {
    final confidenceText = item.confidence != null
        ? '${(item.confidence! * 100).toStringAsFixed(1)}%'
        : 'N/A';

    return ClipRRect(
      borderRadius: BorderRadius.circular(20),
      child: Container(
        height: 220,
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(20),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.15),
              blurRadius: 18,
              offset: const Offset(0, 10),
            ),
          ],
        ),
        child: Stack(
          fit: StackFit.expand,
          children: [
            Image(
              image: item.imageUrl.isNotEmpty
                  ? NetworkImage(item.imageUrl)
                  : const AssetImage('assets/images/leaves.jpg') as ImageProvider,
              fit: BoxFit.cover,
              width: double.infinity,
              errorBuilder: (_, __, ___) => Image.asset(
                'assets/images/leaves.jpg',
                fit: BoxFit.cover,
                width: double.infinity,
              ),
            ),
            Positioned(
              left: 0,
              right: 0,
              bottom: 0,
              child: Container(
                padding: const EdgeInsets.fromLTRB(12, 10, 12, 10),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.bottomCenter,
                    end: Alignment.topCenter,
                    colors: [
                      Colors.black.withOpacity(0.7),
                      Colors.black.withOpacity(0.05),
                    ],
                  ),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Expanded(
                      child: Text(
                        'AI: ${item.title}',
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 14,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ),
                    const SizedBox(width: 10),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(999),
                      ),
                      child: Text(
                        'Confidence: $confidenceText',
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
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
  String _selectedType = 'all';

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
      final logs = await api.getClassificationLogs(binId: widget.binId, limit: 30);
      final items = logs
          .map((e) => TrashHistoryItem(
                imageUrl: ((e['imageUrl'] ?? e['image_url']) ?? '').toString(),
                title: ((e['classificationResult'] ?? e['classification_result']) ?? 'Unknown').toString(),
                  confidence: _toDouble(e['confidenceScore'] ?? e['confidence_score']),
              ))
          .toList();
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

  List<TrashHistoryItem> get _filteredItems {
    if (_selectedType == 'all') return _items;

    return _items.where((item) {
      final type = item.title.toLowerCase().trim();

      switch (_selectedType) {
        case 'organic':
          return type == 'organic';
        case 'recycle':
          return type == 'recycle';
        case 'non_recycle':
          return type == 'nonrecycle' || type == 'non_recycle';
        case 'hazardous':
          return type == 'hazardous';
        default:
          return true;
      }
    }).toList();
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

      body: _loading
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
          : Column(
        children: [
          Container(
            margin: const EdgeInsets.fromLTRB(16, 16, 16, 8),
            padding: const EdgeInsets.symmetric(horizontal: 14),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(16),
            ),
            child: DropdownButtonHideUnderline(
              child: DropdownButton<String>(
                value: _selectedType,
                isExpanded: true,
                items: const [
                  DropdownMenuItem(
                    value: 'all',
                    child: Text('All Waste Types'),
                  ),
                  DropdownMenuItem(
                    value: 'organic',
                    child: Text('Organic Waste'),
                  ),
                  DropdownMenuItem(
                    value: 'recycle',
                    child: Text('Recycle Waste'),
                  ),
                  DropdownMenuItem(
                    value: 'non_recycle',
                    child: Text('Non Recycle Waste'),
                  ),
                  DropdownMenuItem(
                    value: 'hazardous',
                    child: Text('Hazardous Waste'),
                  ),
                ],
                onChanged: (value) {
                  if (value == null) return;
                  setState(() => _selectedType = value);
                },
              ),
            ),
          ),
          Expanded(
            child: _filteredItems.isEmpty
                ? const Center(child: Text('No matching classification images.'))
                : ListView.separated(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
              itemCount: _filteredItems.length,
              separatorBuilder: (_, __) => const SizedBox(height: 18),
              itemBuilder: (context, i) => _HistoryCard(item: _filteredItems[i]),
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
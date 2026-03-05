import 'package:flutter/material.dart';

class HistoryScreen extends StatelessWidget {
  const HistoryScreen({super.key, required this.binId});
  final String binId;

  static const primaryDark = Color(0xFF1F5D35);

  @override
  Widget build(BuildContext context) {
    final items = <TrashHistoryItem>[
      const TrashHistoryItem(
        image: AssetImage('assets/images/trash_1.jpg'),
      ),
      const TrashHistoryItem(
        image: AssetImage('assets/images/trash_2.jpg'),
      ),
      const TrashHistoryItem(
        image: AssetImage('assets/images/trash_3.jpg'),
      ),
    ];

    return Scaffold(
      backgroundColor: const Color(0xFFEAF6EE), // nền mới

      appBar: AppBar(
        backgroundColor: const Color(0xFF2E7D32), 
        elevation: 0,
        foregroundColor: Colors.white, // 
        title: Text(
          'Trash history',
          style: const TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.w700,
          ),
        ),
      ),

      body: ListView.separated(
        padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
        itemCount: items.length,
        separatorBuilder: (_, __) => const SizedBox(height: 18),
        itemBuilder: (context, i) => _HistoryCard(item: items[i]),
      ),
    );
  }
}

class TrashHistoryItem {
  final ImageProvider image;

  const TrashHistoryItem({
    required this.image,
  });
}

class _HistoryCard extends StatelessWidget {
  const _HistoryCard({required this.item});
  final TrashHistoryItem item;

  @override
  Widget build(BuildContext context) {
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
        child: Image(
          image: item.image,
          fit: BoxFit.cover,
          width: double.infinity,
          errorBuilder: (_, __, ___) => const Center(
            child: Icon(Icons.image_not_supported, size: 40),
          ),
        ),
      ),
    );
  }
}
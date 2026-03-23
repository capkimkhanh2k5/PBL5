import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'bin_detail_screen.dart';
import 'ai_chat_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final TextEditingController _searchCtrl = TextEditingController();
  String _query = '';

  // Data mẫu (b1,b2,b3 = TC-01, TC-02, TC-03...) -> bạn thêm ở đây
  final List<TrashCanItem> _items = const [
    TrashCanItem(id: "TC-01", percent: 0.68, lastEmptiedText: "Last emptied: 2 days ago"),
    TrashCanItem(id: "TC-02", percent: 0.42, lastEmptiedText: "Last emptied: 1 day ago"),
    TrashCanItem(id: "TC-03", percent: 0.90, lastEmptiedText: "Last emptied: 5 hours ago"),
    TrashCanItem(id: "TC-04", percent: 0.15, lastEmptiedText: "Last emptied: today"),
  ];

  @override
  void dispose() {
    _searchCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    const bg = Colors.white;
    const headerH = 210.0;

    final filtered = _items.where((e) {
      final q = _query.trim().toLowerCase();
      if (q.isEmpty) return true;
      return e.id.toLowerCase().contains(q);
    }).toList();

    return Scaffold(
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
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  const Text(
                                    "Hello, Hellen!",
                                    style: TextStyle(
                                      color: Colors.white,
                                      fontSize: 30,
                                      fontWeight: FontWeight.w800,
                                    ),
                                  ),
                                  const SizedBox(height: 6),
                                  Row(
                                    children: [
                                      Icon(Icons.cloud_outlined, color: Colors.white.withOpacity(0.9), size: 18),
                                      const SizedBox(width: 6),
                                      Text(
                                        "Sun Cloudy 22°",
                                        style: TextStyle(color: Colors.white.withOpacity(0.9), fontSize: 15),
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
                                showDialog(
                                  context: context,
                                  builder: (context) {
                                    return AlertDialog(
                                      shape: RoundedRectangleBorder(
                                        borderRadius: BorderRadius.circular(16),
                                      ),
                                      title: const Text(
                                        "Logout",
                                        style: TextStyle(fontWeight: FontWeight.bold),
                                      ),
                                      content: const Padding(
                                        padding: EdgeInsets.only(top: 10),
                                        child: Text(
                                          "Are you sure you want to log out?",
                                          style: TextStyle(
                                            fontSize: 15,
                                            fontWeight: FontWeight.w500,
                                            color: Colors.black87,
                                          ),
                                        ),
                                      ),
                                      actions: [
                                        TextButton(
                                          onPressed: () {
                                            Navigator.pop(context); // close dialog
                                          },
                                          child: const Text("Cancel",
                                            style: TextStyle(
                                              fontSize: 17,
                                              color: Colors.black,
                                            ),
                                          ),
                                        ),
                                        TextButton(
                                          onPressed: () async {
                                            Navigator.pop(context);
                                            await FirebaseAuth.instance.signOut();
                                          },
                                          child: const Text(
                                            "Logout",
                                            style: TextStyle(
                                              fontSize: 17,
                                              color: Color(0xFF2F6B3D),
                                              fontWeight: FontWeight.w700,
                                            ),
                                          ),
                                        ),
                                      ],
                                    );
                                  },
                                );
                              },
                              child: Container(
                                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                                decoration: BoxDecoration(
                                  color: Colors.white.withOpacity(0.2),
                                  borderRadius: BorderRadius.circular(20),
                                  border: Border.all(color: Colors.white.withOpacity(0.4)),
                                ),
                                child: Row(
                                  children: [
                                    const Icon(Icons.logout, color: Colors.white, size: 18),
                                    const SizedBox(width: 6),
                                    const Text(
                                      "Logout",
                                      style: TextStyle(
                                        color: Colors.white,
                                        fontWeight: FontWeight.w700,
                                        fontSize: 13,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
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
              const AskAiCard(),

              const SizedBox(height: 16),

              const Text(
                "All",
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.w900),
              ),

              const SizedBox(height: 10),

              // List Trash can cards
              ListView.separated(
                itemCount: filtered.length,
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                separatorBuilder: (_, __) => const SizedBox(height: 12),
                itemBuilder: (context, index) {
                  final it = filtered[index];
                  return TrashCanCard(item: it);
                },
              ),
            ],
          ),
        ),
      ),
    );
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
  const AskAiCard({super.key});

  @override
  Widget build(BuildContext context) {
    const bg = Color(0xFFEAF6C8);

    return InkWell(
      borderRadius: BorderRadius.circular(18),
      onTap: () {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (_) => const AiChatScreen(),
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
  const TrashCanCard({super.key, required this.item});

  final TrashCanItem item;

  @override
  Widget build(BuildContext context) {
    const green = Color(0xFF2F6B3D);
    const plantTile = Color(0xFFB9D98A);

    final pctText = "${(item.percent * 100).round()}%";

    return InkWell(
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
                        fontSize: 18, fontWeight: FontWeight.w900),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    item.lastEmptiedText,
                    style: TextStyle(
                        color: Colors.black.withOpacity(0.65),
                        fontWeight: FontWeight.w600),
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
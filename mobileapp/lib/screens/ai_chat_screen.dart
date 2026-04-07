import 'package:flutter/material.dart';

class AiChatScreen extends StatefulWidget {
  const AiChatScreen({super.key});

  @override
  State<AiChatScreen> createState() => _AiChatScreenState();
}

class _AiChatScreenState extends State<AiChatScreen> {
  final _ctrl = TextEditingController();

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  static const bg = Color(0xFFEAF6EE);
  static const bubble = Colors.white;
  static const chipBg = Colors.white;
  static const primary = Color(0xFF2F6B3D);

  @override
  Widget build(BuildContext context) {
    final top = MediaQuery.of(context).padding.top;

    return Scaffold(
      backgroundColor: bg,
      body: SafeArea(
        child: Stack(
          children: [
            // nền vòng tròn mờ
            Positioned(
              top: -120,
              left: -80,
              child: _softCircle(260),
            ),
            Positioned(
              top: -40,
              right: -90,
              child: _softCircle(240),
            ),

            Column(
              children: [
                // Top bar
                Padding(
                  padding: const EdgeInsets.fromLTRB(12, 6, 12, 6),
                  child: Row(
                    children: [
                      if (Navigator.canPop(context))
                        _circleIcon(
                          icon: Icons.arrow_back,
                          onTap: () => Navigator.pop(context),
                        )
                      else
                        const SizedBox(width: 36),
                      const Spacer(),
                    ],
                  ),
                ),

                // Content
                Expanded(
                  child: SingleChildScrollView(
                    padding: const EdgeInsets.fromLTRB(16, 28, 16, 120),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.center,
                      children: [
                        const Text(
                          "AI Chat",
                          style: TextStyle(
                            fontSize: 22,
                            fontWeight: FontWeight.w900,
                            color: Colors.black87,
                          ),
                        ),
                        const SizedBox(height: 10),
                        const Text(
                          "Good Morning, Zhongli!\nHow can I help you today?",
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            fontSize: 16,
                            height: 1.3,
                            fontWeight: FontWeight.w800,
                            color: Colors.black87,
                          ),
                        ),
                        const SizedBox(height: 10),
                        Text(
                          "Choose a prompt below or write your own to\nstart chatting with your AI assistant!",
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            fontSize: 12.8,
                            height: 1.3,
                            color: Colors.black.withOpacity(0.65),
                            fontWeight: FontWeight.w600,
                          ),
                        ),

                        const SizedBox(height: 16),

                        // Example prompts
                        Align(
                          alignment: Alignment.centerLeft,
                          child: Text(
                            "🌿 Examples",
                            style: TextStyle(
                              fontSize: 13,
                              fontWeight: FontWeight.w800,
                              color: Colors.black.withOpacity(0.70),
                            ),
                          ),
                        ),
                        const SizedBox(height: 10),

                        _promptGrid(context),

                        const SizedBox(height: 18),

                        // Sample chat bubbles (giống ảnh)
                        Align(
                          alignment: Alignment.centerRight,
                          child: _bubble(
                            "Good Morning, Zhongli!\nHow can I help you today?",
                            alignRight: true,
                          ),
                        ),
                        const SizedBox(height: 10),
                        Align(
                          alignment: Alignment.centerLeft,
                          child: _bubble(
                            "Show bins that are nearly full",
                            alignRight: false,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),

            // Bottom input bar + action buttons
            Positioned(
              left: 0,
              right: 0,
              bottom: 0,
              child: SafeArea(
                top: false,
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(14, 10, 14, 40),
                  child: Row(
                    children: [
                      Expanded(
                        child: _input(),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _promptGrid(BuildContext context) {
    // 2 cột giống ảnh
    return LayoutBuilder(
      builder: (_, c) {
        final w = (c.maxWidth - 12) / 2;
        return Wrap(
          spacing: 12,
          runSpacing: 12,
          children: [
            _promptChip(
              width: w,
              icon: Icons.sick_outlined,
              text: "Show bins that\nare nearly full",
            ),
            _promptChip(
              width: w,
              icon: Icons.water_drop_outlined,
              text: "View today’s collection\nschedule",
            ),
            _promptChip(
              width: w,
              icon: Icons.event_note_outlined,
              text: "Track disposal history",
            ),
            _promptChip(
              width: w,
              icon: Icons.add_circle_outline,
              text: "Optimize collection route",
              highlight: true,
            ),
          ],
        );
      },
    );
  }

  Widget _promptChip({
    required double width,
    required IconData icon,
    required String text,
    bool highlight = false,
  }) {
    return InkWell(
      borderRadius: BorderRadius.circular(16),
      onTap: () {
        // demo: đổ text vào input
        _ctrl.text = text.replaceAll('\n', ' ');
        setState(() {});
      },
      child: Container(
        width: width,
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
        decoration: BoxDecoration(
          color: highlight ? bubble : chipBg,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: Colors.white.withOpacity(0.9),
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.06),
              blurRadius: 12,
              offset: const Offset(0, 8),
            ),
          ],
        ),
        child: Row(
          children: [
            Container(
              width: 30,
              height: 30,
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.85),
                shape: BoxShape.circle,
              ),
              child: Icon(icon, size: 18, color: primary),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                text,
                style: const TextStyle(
                  fontSize: 12.5,
                  height: 1.15,
                  fontWeight: FontWeight.w800,
                  color: Colors.black87,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _bubble(String text, {required bool alignRight}) {
    return Container(
      constraints: const BoxConstraints(maxWidth: 280),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      decoration: BoxDecoration(
        color: bubble,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Text(
        text,
        style: const TextStyle(
          fontSize: 13.5,
          height: 1.25,
          fontWeight: FontWeight.w700,
          color: Colors.black87,
        ),
      ),
    );
  }

  Widget _input() {
    return Container(
      height: 46,
      padding: const EdgeInsets.symmetric(horizontal: 12),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.90),
        borderRadius: BorderRadius.circular(26),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.06),
            blurRadius: 12,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Row(
        children: [
          Expanded(
            child: TextField(
              controller: _ctrl,
              decoration: const InputDecoration(
                hintText: "Send a message",
                border: InputBorder.none,
              ),
              onSubmitted: (_) => FocusScope.of(context).unfocus(),
            ),
          ),
          if (_ctrl.text.isNotEmpty)
            GestureDetector(
              onTap: () {
                _ctrl.clear();
                setState(() {});
              },
              child: Icon(Icons.close, size: 18, color: Colors.black54),
            ),
        ],
      ),
    );
  }

  Widget _roundAction(IconData icon, {required VoidCallback onTap}) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(999),
      child: Container(
        width: 44,
        height: 44,
        decoration: BoxDecoration(
          color: primary,
          shape: BoxShape.circle,
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.14),
              blurRadius: 14,
              offset: const Offset(0, 10),
            ),
          ],
        ),
        child: Icon(icon, color: Colors.white, size: 22),
      ),
    );
  }

  Widget _circleIcon({required IconData icon, required VoidCallback onTap}) {
    return InkWell(
      borderRadius: BorderRadius.circular(999),
      onTap: onTap,
      child: Container(
        width: 36,
        height: 36,
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.55),
          shape: BoxShape.circle,
        ),
        child: Icon(icon, color: Colors.black87),
      ),
    );
  }

  Widget _softCircle(double size) {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.35),
        shape: BoxShape.circle,
      ),
    );
  }
}
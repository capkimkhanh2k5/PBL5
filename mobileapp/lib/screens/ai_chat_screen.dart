import 'package:flutter/material.dart';
import '../services/groq_service.dart';
import 'dart:convert';
import '../services/api_service.dart';

class AiChatScreen extends StatefulWidget {
  final ApiService apiService;

  const AiChatScreen({
    super.key,
    required this.apiService,
  });

  @override
  State<AiChatScreen> createState() => _AiChatScreenState();
}

class _AiChatScreenState extends State<AiChatScreen> {

  bool isLoading = false;
  final _scrollCtrl = ScrollController();
  final groq = GroqService();
  final List<Map<String, String>> messages = [];
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
                    controller: _scrollCtrl,
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
                          "Good Morning!\nHow can I help you today?",
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w800,
                          ),
                        ),

                        const SizedBox(height: 16),

                        _promptGrid(context),

                        const SizedBox(height: 20),

                        /// 🔥 CHAT THẬT
                        ...messages.map((msg) {
                          final isUser = msg["role"] == "user";

                          return Padding(
                            padding: const EdgeInsets.only(bottom: 10),
                            child: Align(
                              alignment: isUser
                                  ? Alignment.centerRight
                                  : Alignment.centerLeft,
                              child: _bubble(
                                msg["text"]!,
                                alignRight: isUser,
                              ),
                            ),
                          );
                        }).toList(),
                        if (isLoading)
                          const Padding(
                            padding: EdgeInsets.only(bottom: 10),
                            child: Align(
                              alignment: Alignment.centerLeft,
                              child: Text("AI is typing..."),
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
          final sendText = text.replaceAll('\n', ' ');

          _ctrl.text = sendText;

          _ctrl.selection = TextSelection.fromPosition(
            TextPosition(offset: _ctrl.text.length),
          );

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
            ),
          ),

          /// ❗ NÚT SEND
          IconButton(
            icon: const Icon(Icons.send),
              onPressed: () async {
                final text = _ctrl.text.trim();
                if (text.isEmpty) return;

                setState(() {
                  messages.add({"role": "user", "text": text});
                  _ctrl.clear();
                  isLoading = true;
                });

                _scrollToBottom();

                try {
                  final aiContext = await widget.apiService.getAiContext();
                  print("AI CONTEXT = ${jsonEncode(aiContext)}");
                  print("PICKUP SCHEDULE = ${jsonEncode(aiContext["pickupSchedule"])}");
                  print("PICKUP SCHEDULE TYPE = ${aiContext["pickupSchedule"].runtimeType}");

                  final schedule = aiContext["pickupSchedule"] as List? ?? [];
                  final bins = aiContext["bins"] as List? ?? [];

                  final prompt = """
You are an AI assistant for a Smart Trash App.

You MUST follow ALL rules strictly.

========================
TIME CLASSIFICATION RULES
========================

- If predictedInHours <= 24 → TODAY
- If 24 < predictedInHours <= 48 → TOMORROW
- If predictedInHours > 48 → IGNORE

NEVER mislabel time.

========================
INTENT DETECTION
========================

User intents:

1. "nearly full"
→ Filter bins where avgFill >= 80
→ If none:
   "No bins are nearly full."
→ If multiple:
   → list ALL BIN_ID

2. "today"
→ Filter bins with predictedInHours <= 24
→ If none:
   "No collection scheduled today."
→ If multiple:
   → list ALL bins sorted by predictedInHours (ascending)

3. "tomorrow"
→ Filter bins with 24 < predictedInHours <= 48
→ If none:
   "No collection scheduled tomorrow."
→ If multiple:
   → list ALL bins sorted by predictedInHours (ascending)

4. "history"
→ Respond:
"Disposal history is coming soon."

5. "optimize"
→ Respond:
"Route optimization is coming soon."

6. Unknown
→ Respond:
"I can only help with trash-related features."

========================
RESPONSE FORMAT (STRICT)
========================

Today:
"Today pickups:
- BIN_ID (in Xh)
- BIN_ID (in Xh)"

Tomorrow:
"Tomorrow pickups:
- BIN_ID (in Xh)
- BIN_ID (in Xh)"

Rules:
- Keep it SHORT
- DO NOT explain
- Always sort by predictedInHours (smallest first)

========================
DATA
========================

Pickup Schedule:
${jsonEncode(schedule)}

========================
USER QUESTION
========================

$text
""";
                  final reply = await groq.send(prompt);

                  setState(() {
                    messages.add({"role": "ai", "text": reply});
                    isLoading = false;
                  });

                } catch (e) {
                  setState(() {
                    messages.add({
                      "role": "ai",
                      "text": "Cannot load bin data right now."
                    });
                    isLoading = false;
                  });
                }

                _scrollToBottom();
              }
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
  void _scrollToBottom() {
    Future.delayed(const Duration(milliseconds: 100), () {
      if (_scrollCtrl.hasClients) {
        _scrollCtrl.animateTo(
          _scrollCtrl.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }
}

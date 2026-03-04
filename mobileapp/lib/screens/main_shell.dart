import 'package:flutter/material.dart';
import 'home_screen.dart';
import 'scan_qr_screen.dart';
import 'ai_chat_screen.dart';
import 'schedule_screen.dart';

class MainShell extends StatefulWidget {
  const MainShell({super.key});

  @override
  State<MainShell> createState() => _MainShellState();
}

class _MainShellState extends State<MainShell> {
  int _index = 0;

  final _pages = const [
    HomeScreen(),        // 0
    ScheduleScreen(),       // 1 Schedule
    SizedBox.shrink(),   // 2 Scan (FAB)
    AiChatScreen(),       // 3 AI
    Placeholder(),       // 4 Maps
  ];

  static const _active = Color(0xFF2F6B3D);
  static const _inactive = Colors.black45;

  @override
  Widget build(BuildContext context) {
    final keyboardOpen = MediaQuery.of(context).viewInsets.bottom > 0;

    return Scaffold(
      // bỏ resizeToAvoidBottomInset: false đi
      body: IndexedStack(
        index: _index,
        children: _pages,
      ),

      floatingActionButtonLocation: FloatingActionButtonLocation.centerDocked,
      floatingActionButton: keyboardOpen
          ? null
          : FloatingActionButton(
        backgroundColor: const Color(0xFF2F6B3D), // xanh như cũ
        onPressed: () {
          setState(() => _index = 2);
          Navigator.push(
            context,
            MaterialPageRoute(builder: (_) => const ScanQrScreen()),
          );
        },
        child: const Icon(Icons.qr_code_scanner, color: Colors.white),
      ),

      bottomNavigationBar: keyboardOpen ? null : _buildBottomBar(),
    );
  }

  Widget _buildBottomBar() {
    return BottomAppBar(
      shape: const CircularNotchedRectangle(),
      notchMargin: 8,
      child: SafeArea(
        child: SizedBox(
          height: 60,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _navIcon(Icons.home_rounded, 0),
              _navIcon(Icons.calendar_month, 1),
              const SizedBox(width: 40),
              _navIcon(Icons.smart_toy, 3),
              _navIcon(Icons.map_rounded, 4),
            ],
          ),
        ),
      ),
    );
  }

  Widget _navIcon(IconData icon, int index) {
    final selected = _index == index;

    return IconButton(
      padding: EdgeInsets.zero,
      constraints: const BoxConstraints(),
      onPressed: () => setState(() => _index = index),
      icon: Icon(
        icon,
        size: 26,
        color: selected ? _active : _inactive,
      ),
    );
  }
}
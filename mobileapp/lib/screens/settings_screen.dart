import 'package:flutter/material.dart';
import 'change_password_screen.dart';
import '../services/api_service.dart';
import '../services/auth_service.dart';
import 'package:firebase_auth/firebase_auth.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final AuthService _authService = AuthService();
  late final ApiService _api = ApiService(authService: _authService);

  bool _isLoading = true;
  String _email = '';
  String _username = '';

  @override
  void initState() {
    super.initState();
    _loadProfile();
  }

  Future<void> _loadProfile() async {
    try {
      final data = await _api.getMyProfile();
      if (!mounted) return;
      setState(() {
        _email = (data['email'] ?? '').toString();
        _username = (data['username'] ?? '').toString();
        _isLoading = false;
      });
    } catch (e) {
      setState(() => _isLoading = false);
    }
  }

  // ================= USERNAME =================
  Future<void> _changeUsername() async {
    final controller = TextEditingController(text: _username);

    await showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text("Change Username"),
        content: TextField(
          controller: controller,
          decoration: const InputDecoration(
            hintText: "Enter new username",
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text("Cancel"),
          ),
          ElevatedButton(
            onPressed: () async {
              final newName = controller.text.trim();
              if (newName.isEmpty) return;

              await _api.updateUsername(newName);

              setState(() => _username = newName);

              if (!mounted) return;
              Navigator.pop(context);

              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text("Updated username")),
              );
            },
            child: const Text("Save"),
          ),
        ],
      ),
    );
  }

  // ================= PASSWORD =================
  Future<void> _changePassword() async {
    await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => ChangePasswordScreen(email: _email),
      ),
    );
  }

  // ================= UI =================
  Widget _profileCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      margin: const EdgeInsets.only(bottom: 16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            blurRadius: 10,
            color: Colors.black.withOpacity(0.05),
          )
        ],
      ),
      child: Row(
        children: [
          const CircleAvatar(
            radius: 30,
            child: Icon(Icons.person, size: 30),
          ),
          const SizedBox(width: 12),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                _username,
                style: const TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 16,
                ),
              ),
              Text(_email),
            ],
          )
        ],
      ),
    );
  }

  Widget _item(IconData icon, String title, VoidCallback onTap) {
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      child: ListTile(
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(18),
        ),
        tileColor: Colors.white,
        leading: Icon(icon),
        title: Text(title),
        trailing: const Icon(Icons.chevron_right),
        onTap: onTap,
      ),
    );
  }

  Widget _logoutBtn() {
    return Center(
      child: InkWell(
        onTap: () {
          showDialog(
            context: context,
            builder: (context) {
              return AlertDialog(
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16),
                ),
                title: const Text("Logout"),
                content: const Text("Are you sure you want to log out?"),
                actions: [
                  TextButton(
                    onPressed: () => Navigator.pop(context),
                    child: const Text(
                      "Cancel",
                      style: TextStyle(color: Colors.black),
                    ),
                  ),
                  TextButton(
                    onPressed: () async {
                      Navigator.pop(context);
                      Navigator.pop(context);// đóng dialog
                      await _authService.signOut(); // giống HomeScreen
                    },
                    child: const Text(
                      "Logout",
                      style: TextStyle(
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
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
          decoration: BoxDecoration(
            color: Colors.green.shade700,
            borderRadius: BorderRadius.circular(20),
          ),
          child: const Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.logout, color: Colors.white),
              SizedBox(width: 6),
              Text(
                "Logout",
                style: TextStyle(color: Colors.white),
              ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey.shade100,
      appBar: AppBar(title: const Text("Settings")),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            _profileCard(),
            _item(Icons.person, "Change Username", _changeUsername),
            _item(Icons.lock, "Change Password", _changePassword),
            const SizedBox(height: 20),
            _logoutBtn(),
          ],
        ),
      ),
    );
  }
}
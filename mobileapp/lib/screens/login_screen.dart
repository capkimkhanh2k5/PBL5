import 'package:flutter/material.dart';
import '../utils/top_toast.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import '../services/auth_service.dart';
import '../services/api_service.dart';
import 'register_screen.dart';
import 'forgot_password_screen.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  static const _rememberKey = 'remember_me';
  static const _rememberedEmailKey = 'remembered_email';

  final _emailCtrl = TextEditingController();
  final _passCtrl = TextEditingController();
  final _authService = AuthService();
  final _storage = const FlutterSecureStorage();

  bool _rememberMe = false;
  bool _obscure = true;
  bool _isEmailLoading = false;
  bool _isGoogleLoading = false;

  bool get _isAnyLoading => _isEmailLoading || _isGoogleLoading;

  @override
  void initState() {
    super.initState();
    _loadRememberedLogin();
  }

  @override
  void dispose() {
    _emailCtrl.dispose();
    _passCtrl.dispose();
    super.dispose();
  }

  Future<void> _handleLogin() async {
    final email = _emailCtrl.text.trim();
    final password = _passCtrl.text.trim();

    if (email.isEmpty || password.isEmpty) {
      _showSnackBar('Please enter your email and password.');
      return;
    }

    setState(() => _isEmailLoading = true);

    try {
      await _authService.signInWithEmail(email: email, password: password);
      await _persistRememberState(email);

      // Sync user lên backend
      try {
        final apiService = ApiService(authService: _authService);
        await apiService.syncUser();
      } catch (_) {
        // Backend sync thất bại không chặn login — user vẫn vào app
      }

      if (!mounted) return;
      _completeLoginSuccess();
    } on FirebaseAuthException catch (e) {
      if (!mounted) return;
      _showSnackBar(AuthService.getErrorMessage(e));
    } catch (e) {
      if (!mounted) return;
      _showSnackBar('An error occurred. Please try again.');
    } finally {
      if (mounted) setState(() => _isEmailLoading = false);
    }
  }

  Future<void> _handleGoogleLogin() async {
    setState(() => _isGoogleLoading = true);

    try {
      final credential = await _authService.signInWithGoogle();
      final signedEmail = credential.user?.email?.trim() ?? _emailCtrl.text.trim();
      await _persistRememberState(signedEmail);

      try {
        final apiService = ApiService(authService: _authService);
        await apiService.syncUser();
      } catch (_) {
        // Backend sync thất bại không chặn login — user vẫn vào app
      }

      if (!mounted) return;
      _completeLoginSuccess();
    } on FirebaseAuthException catch (e) {
      if (!mounted) return;
      _showSnackBar(AuthService.getErrorMessage(e));
    } catch (_) {
      if (!mounted) return;
      _showSnackBar('Could not sign in with Google. Please try again.');
    } finally {
      if (mounted) setState(() => _isGoogleLoading = false);
    }
  }

  Future<void> _loadRememberedLogin() async {
    final remembered = await _storage.read(key: _rememberKey);
    final rememberedEmail = await _storage.read(key: _rememberedEmailKey);

    if (!mounted) return;
    setState(() {
      _rememberMe = remembered == 'true';
      if (_rememberMe && rememberedEmail != null && rememberedEmail.isNotEmpty) {
        _emailCtrl.text = rememberedEmail;
      }
    });
  }

  Future<void> _persistRememberState(String email) async {
    if (_rememberMe) {
      await _storage.write(key: _rememberKey, value: 'true');
      await _storage.write(key: _rememberedEmailKey, value: email);
      return;
    }

    await _storage.write(key: _rememberKey, value: 'false');
    await _storage.delete(key: _rememberedEmailKey);
  }

  Future<void> _toggleRemember(bool value) async {
    setState(() => _rememberMe = value);

    if (!value) {
      await _storage.write(key: _rememberKey, value: 'false');
      await _storage.delete(key: _rememberedEmailKey);
    }
  }

  void _completeLoginSuccess() {
    // Nếu LoginScreen đang là route phụ, quay về route gốc để AuthGate hiển thị MainShell.
    Navigator.of(context).popUntil((route) => route.isFirst);
  }

  void _showSnackBar(String message) {
    TopToast.show(context, message);
  }

  @override
  Widget build(BuildContext context) {
    final topImageHeight = 260.0;
    final cardTopRadius = 34.0;

    return Scaffold(
      backgroundColor: const Color(0xFFF3F5F7),
      body: SafeArea(
        child: Stack(
          fit: StackFit.expand, // QUAN TRỌNG: cho Stack full màn hình
          children: [
            // Top image
            Positioned(
              top: 0,
              left: 0,
              right: 0,
              height: topImageHeight,
              child: Image.asset(
                'assets/images/leaves.jpg',
                // 'assets/images/smartTrash.jpg',
                fit: BoxFit.cover,
              ),
            ),

            // White card
            Positioned(
              top: topImageHeight - 40,
              left: 0,
              right: 0,
              bottom: 0,
              child: Container(
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.only(
                    topLeft: Radius.circular(cardTopRadius),
                    topRight: Radius.circular(cardTopRadius),
                  ),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black12,
                      blurRadius: 18,
                      offset: Offset(0, -2),
                    ),
                  ],
                ),
                child: SingleChildScrollView(
                  padding: const EdgeInsets.fromLTRB(20, 26, 20, 24),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Welcome Back',
                        style: TextStyle(
                          fontSize: 26,
                          fontWeight: FontWeight.w800,
                          color: Color(0xFF1F5F3A),
                        ),
                      ),
                      const SizedBox(height: 6),
                      Text(
                        'Login to your account',
                        style: TextStyle(
                          fontSize: 13,
                          color: Colors.black54,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                      const SizedBox(height: 22),
                      _InputField(
                        hint: 'Email',
                        controller: _emailCtrl,
                        prefix: Icons.mail_outline,
                        keyboardType: TextInputType.emailAddress,
                      ),
                      const SizedBox(height: 14),
                      _InputField(
                        hint: 'Password',
                        controller: _passCtrl,
                        prefix: Icons.lock_outline,
                        obscureText: _obscure,
                        suffix: IconButton(
                          onPressed: () => setState(() => _obscure = !_obscure),
                          icon: Icon(
                            _obscure ? Icons.visibility_off : Icons.visibility,
                            color: Colors.black54,
                          ),
                        ),
                      ),
                      const SizedBox(height: 10),
                      Row(
                        children: [
                          Checkbox(
                            value: _rememberMe,
                            onChanged: (v) => _toggleRemember(v ?? false),
                            activeColor: const Color(0xFF2F6B3D),
                          ),
                          const Text('Remember me',
                              style: TextStyle(fontSize: 13)),
                          const Spacer(),
                          TextButton(
                            onPressed: () {
                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                  builder: (_) => const ForgotPasswordScreen(),
                                ),
                              );
                            },
                            child: const Text(
                              'Forgot Password?',
                              style: TextStyle(
                                fontSize: 13,
                                color: Color(0xFF2F6B3D),
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 6),
                      SizedBox(
                        width: double.infinity,
                        height: 54,
                        child: ElevatedButton(
                          onPressed: _isAnyLoading ? null : _handleLogin,
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFF2F6B3D),
                            foregroundColor: Colors.white,
                            elevation: 0,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(28),
                            ),
                          ),
                            child: _isEmailLoading
                              ? const SizedBox(
                                  width: 24,
                                  height: 24,
                                  child: CircularProgressIndicator(
                                    color: Colors.white,
                                    strokeWidth: 2.5,
                                  ),
                                )
                              : const Text(
                                  'Login',
                                  style: TextStyle(
                                      fontSize: 16,
                                      fontWeight: FontWeight.w800),
                                ),
                        ),
                      ),
                      const SizedBox(height: 18),
                      Row(
                        children: [
                          Expanded(
                            child: Divider(
                              color: Colors.black.withOpacity(0.14),
                              thickness: 1,
                            ),
                          ),
                          Padding(
                            padding: const EdgeInsets.symmetric(horizontal: 10),
                            child: Text(
                              'or',
                              style: TextStyle(
                                fontSize: 12,
                                fontWeight: FontWeight.w600,
                                color: Colors.black.withOpacity(0.55),
                              ),
                            ),
                          ),
                          Expanded(
                            child: Divider(
                              color: Colors.black.withOpacity(0.14),
                              thickness: 1,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 14),
                      SizedBox(
                        width: double.infinity,
                        height: 52,
                        child: OutlinedButton(
                          onPressed: _isAnyLoading ? null : _handleGoogleLogin,
                          style: OutlinedButton.styleFrom(
                            backgroundColor: Colors.white,
                            foregroundColor: const Color(0xFF202124),
                            side: BorderSide(
                              color: Colors.black.withOpacity(0.12),
                            ),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(28),
                            ),
                          ),
                            child: _isGoogleLoading
                              ? const SizedBox(
                                  width: 24,
                                  height: 24,
                                  child: CircularProgressIndicator(
                                    color: Color(0xFF2F6B3D),
                                    strokeWidth: 2.5,
                                  ),
                                )
                              : Row(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: const [
                                    CircleAvatar(
                                      radius: 12,
                                      backgroundColor: Color(0xFFF1F3F4),
                                      child: Text(
                                        'G',
                                        style: TextStyle(
                                          fontWeight: FontWeight.w800,
                                          color: Color(0xFF4285F4),
                                          fontSize: 14,
                                        ),
                                      ),
                                    ),
                                    SizedBox(width: 10),
                                    Text(
                                      'Continue with Google',
                                      style: TextStyle(
                                        fontSize: 15,
                                        fontWeight: FontWeight.w700,
                                      ),
                                    ),
                                  ],
                                ),
                        ),
                      ),
                      const SizedBox(height: 16),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Text(
                            "Don’t have an account? ",
                            style: TextStyle(
                              fontSize: 13,
                              color: Colors.black.withOpacity(0.65),
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          InkWell(
                            onTap: () {
                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                    builder: (_) => const RegisterScreen()),
                              );
                            },
                            child: const Text(
                              "Register",
                              style: TextStyle(
                                fontSize: 13,
                                color: Color(0xFF2F6B3D),
                                fontWeight: FontWeight.w800,
                                decoration: TextDecoration.underline,
                              ),
                            ),
                          ),
                        ],
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
}

class _InputField extends StatelessWidget {
  const _InputField({
    required this.hint,
    required this.controller,
    required this.prefix,
    this.suffix,
    this.obscureText = false,
    this.keyboardType,
  });

  final String hint;
  final TextEditingController controller;
  final IconData prefix;
  final Widget? suffix;
  final bool obscureText;
  final TextInputType? keyboardType;

  @override
  Widget build(BuildContext context) {
    return TextField(
      controller: controller,
      obscureText: obscureText,
      keyboardType: keyboardType,
      decoration: InputDecoration(
        hintText: hint,
        hintStyle: TextStyle(color: Colors.black.withOpacity(0.45)),
        prefixIcon: Icon(prefix, color: Colors.black.withOpacity(0.55)),
        suffixIcon: suffix,
        filled: true,
        fillColor: const Color(0xFFF0F3F1),
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 14, vertical: 16),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: BorderSide.none,
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: BorderSide(color: Colors.black.withOpacity(0.06)),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(color: Color(0xFF2F6B3D), width: 1.3),
        ),
      ),
    );
  }
}

import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import '../services/auth_service.dart';

class ForgotPasswordScreen extends StatefulWidget {
  const ForgotPasswordScreen({super.key});

  @override
  State<ForgotPasswordScreen> createState() => _ForgotPasswordScreenState();
}

class _ForgotPasswordScreenState extends State<ForgotPasswordScreen> {
  final _emailCtrl = TextEditingController();
  final _authService = AuthService();
  bool _isLoading = false;
  bool _emailSent = false;

  @override
  void dispose() {
    _emailCtrl.dispose();
    super.dispose();
  }

  Future<void> _handleResetPassword() async {
    final email = _emailCtrl.text.trim();

    if (email.isEmpty) {
      _showSnackBar('Please enter your email address.', isError: true);
      return;
    }

    setState(() => _isLoading = true);

    try {
      await _authService.sendPasswordResetEmail(email: email);
      if (!mounted) return;
      setState(() => _emailSent = true);
    } on FirebaseAuthException catch (e) {
      if (!mounted) return;
      _showSnackBar(AuthService.getErrorMessage(e), isError: true);
    } catch (e) {
      if (!mounted) return;
      _showSnackBar('An error occurred. Please try again.', isError: true);
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  void _showSnackBar(String message, {bool isError = false}) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: isError ? Colors.red.shade700 : const Color(0xFF2F6B3D),
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    const green = Color(0xFF2F6B3D);

    return Scaffold(
      backgroundColor: const Color(0xFFF3F5F7),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new, color: green),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24),
          child: _emailSent ? _buildSuccessView(green) : _buildFormView(green),
        ),
      ),
    );
  }

  Widget _buildFormView(Color green) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const SizedBox(height: 20),

        const Icon(
          Icons.lock_reset_rounded,
          size: 64,
          color: Color(0xFF2F6B3D),
        ),

        const SizedBox(height: 20),

        const Text(
          'Quên mật khẩu?',
          style: TextStyle(
            fontSize: 26,
            fontWeight: FontWeight.w800,
            color: Color(0xFF1F5F3A),
          ),
        ),
        const SizedBox(height: 10),
        Text(
          'Enter your registered email. We will send a password reset link to your email.',
          style: TextStyle(
            fontSize: 14,
            color: Colors.black.withOpacity(0.6),
            height: 1.5,
          ),
        ),

        const SizedBox(height: 30),

        TextField(
          controller: _emailCtrl,
          keyboardType: TextInputType.emailAddress,
          decoration: InputDecoration(
            hintText: 'Email',
            hintStyle: TextStyle(color: Colors.black.withOpacity(0.45)),
            prefixIcon: Icon(Icons.mail_outline, color: Colors.black.withOpacity(0.55)),
            filled: true,
            fillColor: const Color(0xFFF0F3F1),
            contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 16),
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
        ),

        const SizedBox(height: 24),

        SizedBox(
          width: double.infinity,
          height: 54,
          child: ElevatedButton(
            onPressed: _isLoading ? null : _handleResetPassword,
            style: ElevatedButton.styleFrom(
              backgroundColor: green,
              foregroundColor: Colors.white,
              elevation: 0,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(28),
              ),
            ),
            child: _isLoading
                ? const SizedBox(
                    width: 24,
                    height: 24,
                    child: CircularProgressIndicator(
                      color: Colors.white,
                      strokeWidth: 2.5,
                    ),
                  )
                : const Text(
                    'Send Password Reset Link',
                    style: TextStyle(fontSize: 15, fontWeight: FontWeight.w700),
                  ),
          ),
        ),
      ],
    );
  }

  Widget _buildSuccessView(Color green) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(
          Icons.mark_email_read_rounded,
          size: 80,
          color: green,
        ),
        const SizedBox(height: 24),
        const Text(
          'Email sent!',
          style: TextStyle(
            fontSize: 24,
            fontWeight: FontWeight.w800,
            color: Color(0xFF1F5F3A),
          ),
        ),
        const SizedBox(height: 12),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20),
          child: Text(
            'Check your inbox ${_emailCtrl.text.trim()} for a password reset link.\n\nIf you don\'t see the email, please check your Spam folder.',
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 14,
              color: Colors.black.withOpacity(0.6),
              height: 1.5,
            ),
          ),
        ),
        const SizedBox(height: 30),
        SizedBox(
          width: double.infinity,
          height: 54,
          child: ElevatedButton(
            onPressed: () => Navigator.pop(context),
            style: ElevatedButton.styleFrom(
              backgroundColor: green,
              foregroundColor: Colors.white,
              elevation: 0,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(28),
              ),
            ),
            child: const Text(
              'Back to Login',
              style: TextStyle(fontSize: 15, fontWeight: FontWeight.w700),
            ),
          ),
        ),
        const SizedBox(height: 14),
        TextButton(
          onPressed: () {
            setState(() => _emailSent = false);
          },
          child: Text(
            'Resend Email',
            style: TextStyle(
              fontSize: 14,
              color: green,
              fontWeight: FontWeight.w700,
            ),
          ),
        ),
      ],
    );
  }
}

import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

import 'screens/home_screen.dart';

void main() {
  runApp(const DefVoiceApp());
}

class DefVoiceApp extends StatelessWidget {
  const DefVoiceApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'DefVoice',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF09090B),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF6366F1),
          surface: Color(0xFF18181B),
        ),
        inputDecorationTheme: const InputDecorationTheme(
          border: OutlineInputBorder(),
          isDense: true,
        ),
      ),
      home: const _PermissionGate(child: HomeScreen()),
    );
  }
}

/// WebRTC's getUserMedia fails with an opaque platform error if RECORD_AUDIO was
/// never granted, and the failure looks like a networking bug. Ask first.
class _PermissionGate extends StatefulWidget {
  final Widget child;
  const _PermissionGate({required this.child});

  @override
  State<_PermissionGate> createState() => _PermissionGateState();
}

class _PermissionGateState extends State<_PermissionGate> {
  bool _granted = false;
  bool _checked = false;

  @override
  void initState() {
    super.initState();
    _ask();
  }

  Future<void> _ask() async {
    final status = await Permission.microphone.request();
    if (!mounted) return;
    setState(() {
      _granted = status.isGranted;
      _checked = true;
    });
  }

  @override
  Widget build(BuildContext context) {
    if (!_checked) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    if (_granted) return widget.child;
    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(28),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.mic_off, size: 56, color: Colors.white54),
            const SizedBox(height: 16),
            const Text(
              'Microphone access is required to place or receive a call.',
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 20),
            FilledButton(onPressed: _ask, child: const Text('Grant access')),
            const SizedBox(height: 8),
            const TextButton(
              onPressed: openAppSettings,
              child: Text('Open app settings'),
            ),
          ],
        ),
      ),
    );
  }
}

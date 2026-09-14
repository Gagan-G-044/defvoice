import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';

import '../config.dart';
import 'dialer_screen.dart';
import 'receiver_screen.dart';

/// Setup screen. Exists mostly so the LAN address can be fixed on the phone
/// instead of rebuilding the APK when DHCP moves the laptop.
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  late final _host = TextEditingController(text: AppConfig.backendHost);
  late final _port = TextEditingController(text: '${AppConfig.backendPort}');
  late final _session = TextEditingController(text: AppConfig.sessionId);
  late final _token = TextEditingController(text: AppConfig.token);

  String? _healthMsg;
  bool _healthOk = false;
  bool _checking = false;

  @override
  void dispose() {
    _host.dispose();
    _port.dispose();
    _session.dispose();
    _token.dispose();
    super.dispose();
  }

  void _apply() {
    AppConfig.backendHost = _host.text.trim();
    AppConfig.backendPort = int.tryParse(_port.text.trim()) ?? 8000;
    AppConfig.sessionId = _session.text.trim();
    AppConfig.token = _token.text.trim();
  }

  /// dart:io HttpClient rather than the `http` package -- one less dependency,
  /// and this is the only HTTP call the app makes.
  Future<void> _checkHealth() async {
    _apply();
    setState(() {
      _checking = true;
      _healthMsg = null;
    });
    final client = HttpClient()..connectionTimeout = const Duration(seconds: 4);
    try {
      final req = await client.getUrl(Uri.parse('${AppConfig.httpBase}/health'));
      final res = await req.close().timeout(const Duration(seconds: 5));
      final body = await res.transform(utf8.decoder).join();
      final j = jsonDecode(body) as Map<String, dynamic>;
      final degraded = j['degraded'] == true;
      setState(() {
        _healthOk = true;
        _healthMsg = degraded
            ? 'Backend up, but running the STUB countermeasure. '
                'Scores are not meaningful until real weights are loaded.'
            : 'Backend up. Engines loaded.';
      });
    } on TimeoutException {
      setState(() {
        _healthOk = false;
        _healthMsg = 'Timed out. Same Wi-Fi? Venue APs often block '
            'client-to-client traffic -- use a laptop hotspot, or USB: '
            'adb reverse tcp:8000 tcp:8000 and host 127.0.0.1';
      });
    } catch (e) {
      setState(() {
        _healthOk = false;
        _healthMsg = 'Unreachable: $e';
      });
    } finally {
      client.close(force: true);
      if (mounted) setState(() => _checking = false);
    }
  }

  void _go(Widget screen) {
    _apply();
    Navigator.of(context).push(MaterialPageRoute(builder: (_) => screen));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('DefVoice'),
        actions: const [
          Padding(
            padding: EdgeInsets.only(right: 14),
            child: Center(child: Text('SIH26104', style: TextStyle(fontSize: 12))),
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          const Text(
            'Real-time voice clone detection for in-app calls',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 6),
          const Text(
            'Both phones connect to the laptop, which relays the audio and '
            'analyses a copy. Set the laptop address below, then pick a role. '
            'Both phones must use the same session ID and token.',
            style: TextStyle(color: Colors.white70, height: 1.35),
          ),
          const SizedBox(height: 22),
          Row(
            children: [
              Expanded(
                flex: 3,
                child: TextField(
                  controller: _host,
                  keyboardType: TextInputType.url,
                  decoration: const InputDecoration(
                    labelText: 'Server Address',
                    hintText: '10.0.2.2 or <server-ip>',
                    prefixIcon: Icon(Icons.dns_rounded),
                  ),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: TextField(
                  controller: _port,
                  keyboardType: TextInputType.number,
                  decoration: const InputDecoration(
                    labelText: 'Port',
                    hintText: '8000',
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _session,
            decoration: const InputDecoration(
              labelText: 'Session ID',
              hintText: 'call_001',
              prefixIcon: Icon(Icons.tag_rounded),
            ),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _token,
            decoration: const InputDecoration(
              labelText: 'Security Token',
              hintText: 'sih26104-change-me',
              prefixIcon: Icon(Icons.key_rounded),
            ),
          ),
          const SizedBox(height: 16),
          OutlinedButton.icon(
            onPressed: _checking ? null : _checkHealth,
            icon: _checking
                ? const SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.wifi_find),
            label: const Text('Test backend'),
          ),
          if (_healthMsg != null) ...[
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: (_healthOk ? Colors.green : Colors.red).withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(
                  color: _healthOk ? Colors.green : Colors.red,
                ),
              ),
              child: Text(_healthMsg!, style: const TextStyle(height: 1.35)),
            ),
          ],
          const SizedBox(height: 28),
          const Divider(),
          const SizedBox(height: 12),
          const Text('Pick this phone\'s role',
              style: TextStyle(fontWeight: FontWeight.w600)),
          const SizedBox(height: 12),
          _RoleCard(
            icon: Icons.record_voice_over,
            title: 'Phone A -- Caller',
            subtitle:
                'Places the call. Can speak live, or play a cloned clip out of '
                'the loudspeaker so the mic re-captures it. Echo cancellation '
                'is disabled on this phone, so use a headset if you speak live.',
            onTap: () => _go(const DialerScreen()),
          ),
          const SizedBox(height: 12),
          _RoleCard(
            icon: Icons.phone_in_talk,
            title: 'Phone B -- Receiver',
            subtitle:
                'Answers the call and shows the live authenticity gauge, the '
                'reasons behind it, and the step-up prompt.',
            onTap: () => _go(const ReceiverScreen()),
          ),
          const SizedBox(height: 24),
          const Text(
            'This detects synthetic speech on calls placed inside this app. It '
            'does not and cannot intercept cellular, WhatsApp, or dialer calls '
            '-- Android does not permit it.',
            style: TextStyle(fontSize: 12, color: Colors.white54, height: 1.4),
          ),
        ],
      ),
    );
  }
}

class _RoleCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final VoidCallback onTap;

  const _RoleCard({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: EdgeInsets.zero,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(12),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            children: [
              Icon(icon, size: 30),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(title,
                        style: const TextStyle(
                            fontSize: 16, fontWeight: FontWeight.w600)),
                    const SizedBox(height: 4),
                    Text(subtitle,
                        style: const TextStyle(
                            fontSize: 12.5,
                            color: Colors.white70,
                            height: 1.35)),
                  ],
                ),
              ),
              const Icon(Icons.chevron_right),
            ],
          ),
        ),
      ),
    );
  }
}

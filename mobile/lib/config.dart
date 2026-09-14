/// Runtime configuration.
///
/// Change [backendHost] to your laptop's LAN address before every demo -- DHCP
/// will have moved it. `ipconfig` on Windows, `ip addr` on Linux.
///
/// Fallback when the network fights you: USB. Plug each phone in and run
///   adb -s <serial> reverse tcp:8000 tcp:8000
/// then set backendHost to '127.0.0.1'. Works with no Wi-Fi at all, which has
/// saved more demos than any router.
library;

class AppConfig {
  /// Default backend host. Override at compile time with `--dart-define=SERVER_HOST=...`
  /// or at runtime via the connection input in the app.
  static String backendHost = const String.fromEnvironment(
    'SERVER_HOST',
    defaultValue: '', // Starts empty for clean public release
  );
  static int backendPort = const int.fromEnvironment(
    'SERVER_PORT',
    defaultValue: 8000,
  );

  /// Must match DEFVOICE_TOKEN on the backend.
  static String token = const String.fromEnvironment(
    'AUTH_TOKEN',
    defaultValue: 'sih26104-change-me',
  );

  /// Both phones must use the same value to be joined into one call.
  static String sessionId = const String.fromEnvironment(
    'SESSION_ID',
    defaultValue: 'call_001',
  );

  /// Effective host using 10.0.2.2 fallback when input is left blank.
  static String get effectiveHost =>
      backendHost.trim().isEmpty ? '10.0.2.2' : backendHost.trim();

  static String get httpBase => 'http://$effectiveHost:$backendPort';

  static String signalUrl(String role) =>
      'ws://$effectiveHost:$backendPort/ws/signal/$sessionId'
      '?role=$role&token=$token';

  static String telemetryUrl() =>
      'ws://$effectiveHost:$backendPort/ws/telemetry/$sessionId?token=$token';

  /// Clone clips bundled in assets/clones/. First entry is played by default.
  static const List<CloneClip> cloneClips = [
    CloneClip(
      label: 'Wire 8 lakh (Hindi)',
      asset: 'assets/clones/cfo_wire_8lakh_hi.wav',
      transcript: 'आठ लाख रुपये तुरंत इस खाते में भेजो।',
    ),
    CloneClip(
      label: 'Share the OTP (Indian English)',
      asset: 'assets/clones/cfo_otp_en.wav',
      transcript: 'Read me the OTP now, the vendor payment is stuck.',
    ),
    CloneClip(
      label: 'Emergency authorisation (Kannada)',
      asset: 'assets/clones/cfo_auth_kn.wav',
      transcript: 'ತಕ್ಷಣ ಹಣ ಕಳುಹಿಸಿ, ಇದು ತುರ್ತು.',
    ),
  ];
}

class CloneClip {
  final String label;
  final String asset;
  final String transcript;
  const CloneClip({
    required this.label,
    required this.asset,
    required this.transcript,
  });
}

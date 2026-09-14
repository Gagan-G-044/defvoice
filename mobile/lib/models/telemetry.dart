/// Telemetry frames pushed by the backend over `/ws/telemetry/{session}`.
///
/// Parsing is deliberately forgiving. A demo must not crash because a field was
/// renamed on the backend an hour before judging; unknown or missing fields
/// degrade to safe defaults and the UI keeps rendering.
library;

enum ThreatLevel { safe, caution, suspicious, critical }

ThreatLevel threatLevelFromName(String? name) {
  switch ((name ?? '').toUpperCase()) {
    case 'CRITICAL':
      return ThreatLevel.critical;
    case 'SUSPICIOUS':
      return ThreatLevel.suspicious;
    case 'CAUTION':
      return ThreatLevel.caution;
    default:
      return ThreatLevel.safe;
  }
}

String threatLevelLabel(ThreatLevel l) {
  switch (l) {
    case ThreatLevel.critical:
      return 'CRITICAL';
    case ThreatLevel.suspicious:
      return 'SUSPICIOUS';
    case ThreatLevel.caution:
      return 'CAUTION';
    case ThreatLevel.safe:
      return 'SAFE';
  }
}

class RiskFrame {
  /// 0-100. Smoothed, hysteresis-gated risk that the far end is synthetic.
  final double risk;

  /// 100 - risk. Shown to the user because "authenticity" reads better on a
  /// gauge than "risk", but it is the same number.
  final double authenticityPct;

  final ThreatLevel level;

  /// True when the Wav2Vec2 acoustic analysis classifies the audio as
  /// AI-generated (vocoder / neural synthesis artifacts).
  final bool isAiClone;

  /// 0-100 confidence that the audio contains neural synthesis artifacts.
  final int cloneConfidencePct;

  /// Raw per-window CM output before smoothing. Useful on the dashboard, noisy
  /// enough that it should not drive the phone UI.
  final double? pSynthetic;

  /// Named contributions, e.g. {'cm': 71.0, 'asv': 0.0, 'context': 12.0}.
  final Map<String, double> components;

  /// Human-readable justifications, already ordered by weight.
  final List<String> reasons;

  /// Partial ASR text. May be empty; never gates the verdict.
  final String transcript;

  final double? inferenceMs;
  final int droppedWindows;

  /// True when the backend is running the stub countermeasure. The UI must say
  /// so out loud -- a placeholder that looks like a detector is worse than no
  /// detector at all.
  final bool degraded;

  /// Backend's own recommendation that the receiver demand a second factor.
  final bool stepUpRequired;

  RiskFrame({
    required this.risk,
    required this.authenticityPct,
    required this.level,
    this.isAiClone = false,
    this.cloneConfidencePct = 0,
    required this.pSynthetic,
    required this.components,
    required this.reasons,
    required this.transcript,
    required this.inferenceMs,
    required this.droppedWindows,
    required this.degraded,
    required this.stepUpRequired,
  });

  static double _d(dynamic v, [double fallback = 0]) {
    if (v is num) return v.toDouble();
    if (v is String) return double.tryParse(v) ?? fallback;
    return fallback;
  }

  factory RiskFrame.fromJson(Map<String, dynamic> j) {
    final risk = _d(j['risk']);
    final pSyn = j['p_synthetic'] == null ? null : _d(j['p_synthetic']);
    final comps = <String, double>{};
    final rawComps = j['components'];
    if (rawComps is Map) {
      rawComps.forEach((k, v) => comps['$k'] = _d(v));
    }
    final reasons = <String>[];
    final rawReasons = j['reasons'];
    if (rawReasons is List) {
      for (final r in rawReasons) {
        reasons.add('$r');
      }
    }
    return RiskFrame(
      risk: risk,
      authenticityPct: j.containsKey('authenticity_pct')
          ? _d(j['authenticity_pct'], 100 - risk)
          : 100 - risk,
      level: threatLevelFromName(j['level'] as String?),
      isAiClone: j['is_ai_clone'] as bool? ??
          ((pSyn ?? 0.0) >= 0.50),
      cloneConfidencePct: (j['clone_confidence_pct'] as num?)?.toInt() ??
          ((pSyn ?? 0.0) * 100).round(),
      pSynthetic: pSyn,
      components: comps,
      reasons: reasons,
      transcript: (j['transcript'] as String?) ?? '',
      inferenceMs: j['inference_ms'] == null ? null : _d(j['inference_ms']),
      droppedWindows: (j['dropped_windows'] as num?)?.toInt() ?? 0,
      degraded: j['degraded'] == true,
      stepUpRequired: j['step_up_required'] == true,
    );
  }

  static RiskFrame get initial => RiskFrame(
        risk: 0,
        authenticityPct: 100,
        level: ThreatLevel.safe,
        pSynthetic: null,
        components: const {},
        reasons: const [],
        transcript: '',
        inferenceMs: null,
        droppedWindows: 0,
        degraded: false,
        stepUpRequired: false,
      );
}

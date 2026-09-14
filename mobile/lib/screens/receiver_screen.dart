import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../config.dart';
import '../models/telemetry.dart';
import '../services/rtc_service.dart';
import '../services/telemetry_service.dart';
import '../widgets/risk_gauge.dart';

/// Recorded incident of synthetic speech or suspicious activity.
class ThreatIncident {
  final DateTime timestamp;
  final int cloneConfidencePct;
  final int authenticityPct;
  final List<String> reasons;

  ThreatIncident({
    required this.timestamp,
    required this.cloneConfidencePct,
    required this.authenticityPct,
    required this.reasons,
  });
}

/// Phone B. Answers the call and shows the verdict.
///
/// Telemetry connects as soon as this screen opens, before the call is answered.
/// That is deliberate: if the socket only came up on answer, a telemetry problem
/// would be indistinguishable from a call problem, and you would be debugging
/// two things at once in front of judges.
class ReceiverScreen extends StatefulWidget {
  const ReceiverScreen({super.key});

  @override
  State<ReceiverScreen> createState() => _ReceiverScreenState();
}

class _ReceiverScreenState extends State<ReceiverScreen> {
  final _rtc = RtcService();
  final _telemetry = TelemetryService();

  StreamSubscription? _stateSub;
  StreamSubscription? _frameSub;
  StreamSubscription? _statusSub;

  CallState _state = CallState.idle;
  String _telemetryStatus = 'connecting';
  RiskFrame _frame = RiskFrame.initial;
  ThreatLevel _lastLevel = ThreatLevel.safe;
  bool _stepUpDismissed = false;

  // Cumulative threat session tracking
  final List<ThreatIncident> _incidents = [];
  final Set<String> _cumulativeReasons = {};
  int _peakCloneConfidence = 0;
  double _lowestAuthenticity = 100.0;
  ThreatLevel _peakThreatLevel = ThreatLevel.safe;
  bool _hasDetectedClone = false;
  DateTime? _callStartTime;
  DateTime? _callEndTime;
  DateTime? _lastIncidentRecordedAt;
  bool _wasConnected = false;

  @override
  void initState() {
    super.initState();
    _stateSub = _rtc.stateStream.listen((s) {
      if (mounted) {
        if (s == CallState.connected && !_wasConnected) {
          _resetSessionCounters();
          _callStartTime = DateTime.now();
        } else if (_wasConnected && s == CallState.idle) {
          _onCallEnded();
        }
        _wasConnected = (s == CallState.connected);
        setState(() => _state = s);
      }
    });
    _frameSub = _telemetry.frames.listen(_onFrame);
    _statusSub = _telemetry.status.listen((s) {
      if (mounted) setState(() => _telemetryStatus = s);
    });
    _telemetry.connect();
  }

  void _resetSessionCounters() {
    _incidents.clear();
    _cumulativeReasons.clear();
    _peakCloneConfidence = 0;
    _lowestAuthenticity = 100.0;
    _peakThreatLevel = ThreatLevel.safe;
    _hasDetectedClone = false;
    _lastIncidentRecordedAt = null;
    _callEndTime = null;
  }

  void _onFrame(RiskFrame f) {
    if (!mounted) return;

    // Buzz on escalation only.
    if (f.level.index > _lastLevel.index &&
        f.level.index >= ThreatLevel.suspicious.index) {
      HapticFeedback.heavyImpact();
    }

    // Cumulative metrics
    if (f.cloneConfidencePct > _peakCloneConfidence) {
      _peakCloneConfidence = f.cloneConfidencePct;
    }
    if (f.authenticityPct < _lowestAuthenticity) {
      _lowestAuthenticity = f.authenticityPct;
    }
    if (f.level.index > _peakThreatLevel.index) {
      _peakThreatLevel = f.level;
    }
    if (f.isAiClone || f.cloneConfidencePct >= 50) {
      _hasDetectedClone = true;
    }
    _cumulativeReasons.addAll(f.reasons);

    // Record incident on threat occurrence (throttled to avoid flooding)
    final isThreat =
        f.isAiClone || f.cloneConfidencePct >= 50 || f.level == ThreatLevel.critical;
    if (isThreat) {
      final now = DateTime.now();
      if (_lastIncidentRecordedAt == null ||
          now.difference(_lastIncidentRecordedAt!).inMilliseconds >= 2500 ||
          f.cloneConfidencePct > _peakCloneConfidence) {
        _lastIncidentRecordedAt = now;
        _incidents.add(
          ThreatIncident(
            timestamp: now,
            cloneConfidencePct: f.cloneConfidencePct,
            authenticityPct: f.authenticityPct.round(),
            reasons: List<String>.from(f.reasons),
          ),
        );
      }
    }

    setState(() {
      if (f.level.index > _lastLevel.index) _stepUpDismissed = false;
      _lastLevel = f.level;
      _frame = f;
    });
  }

  Future<void> _answer() async {
    try {
      _resetSessionCounters();
      _callStartTime = DateTime.now();
      await _rtc.connect(role: CallRole.receiver);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(_rtc.lastError ?? '$e'),
          duration: const Duration(seconds: 5),
        ),
      );
    }
  }

  Future<void> _hangUp() async {
    await _rtc.hangUp();
    _onCallEnded();
  }

  void _onCallEnded() {
    _callEndTime = DateTime.now();
    if (!mounted) return;
    // Show forensic report modal if an active call took place or any threats were observed
    if (_wasConnected || _incidents.isNotEmpty || _hasDetectedClone) {
      _showPostCallReport();
    }
  }

  void _showPostCallReport() {
    showDialog(
      context: context,
      barrierDismissible: true,
      builder: (ctx) => _PostCallReportDialog(
        sessionId: AppConfig.sessionId,
        callStartTime: _callStartTime,
        callEndTime: _callEndTime,
        incidents: List.unmodifiable(_incidents),
        peakCloneConfidence: _peakCloneConfidence,
        lowestAuthenticity: _lowestAuthenticity,
        peakThreatLevel: _peakThreatLevel,
        hasDetectedClone: _hasDetectedClone,
        cumulativeReasons: List.unmodifiable(_cumulativeReasons),
        onReset: () {
          setState(_resetSessionCounters);
          Navigator.of(ctx).pop();
        },
      ),
    );
  }

  @override
  void dispose() {
    _stateSub?.cancel();
    _frameSub?.cancel();
    _statusSub?.cancel();
    _telemetry.dispose();
    _rtc.dispose();
    super.dispose();
  }

  // -------------------------------------------------------------------------
  @override
  Widget build(BuildContext context) {
    final connected = _state == CallState.connected;
    final connecting = _state == CallState.connecting;
    final showStepUp = _frame.stepUpRequired && !_stepUpDismissed;
    final showCloneBanner = _hasDetectedClone || _frame.isAiClone;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Phone B -- Receiver'),
        actions: [
          if (_incidents.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(right: 8),
              child: GestureDetector(
                onTap: _showPostCallReport,
                child: Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: Colors.redAccent.withValues(alpha: 0.2),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: Colors.redAccent, width: 1),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(Icons.warning_amber_rounded,
                          size: 14, color: Colors.redAccent),
                      const SizedBox(width: 4),
                      Text(
                        '${_incidents.length} threat${_incidents.length == 1 ? '' : 's'}',
                        style: const TextStyle(
                          fontSize: 11,
                          fontWeight: FontWeight.bold,
                          color: Colors.redAccent,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          Padding(
            padding: const EdgeInsets.only(right: 14),
            child: Center(
              child: Text(
                _telemetryStatus,
                style: TextStyle(
                  fontSize: 11,
                  color: _telemetryStatus == 'live'
                      ? Colors.greenAccent
                      : Colors.amberAccent,
                ),
              ),
            ),
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(20, 14, 20, 28),
        children: [
          if (_telemetry.degraded || _frame.degraded) const _DegradedBanner(),

          // Persistent Threat Counter Chip
          if (_incidents.isNotEmpty || _hasDetectedClone) ...[
            GestureDetector(
              onTap: _showPostCallReport,
              child: Container(
                margin: const EdgeInsets.only(bottom: 12),
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                decoration: BoxDecoration(
                  color: _hasDetectedClone
                      ? Colors.red.withValues(alpha: 0.18)
                      : Colors.amber.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(
                    color: _hasDetectedClone
                        ? Colors.redAccent
                        : Colors.amberAccent,
                    width: 1.2,
                  ),
                ),
                child: Row(
                  children: [
                    Icon(
                      Icons.shield_outlined,
                      size: 18,
                      color: _hasDetectedClone
                          ? Colors.redAccent
                          : Colors.amberAccent,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        'Threats Flagged: ${_incidents.length} · Peak: $_peakCloneConfidence% · Min Auth: ${_lowestAuthenticity.toStringAsFixed(0)}%',
                        style: TextStyle(
                          fontSize: 11.5,
                          fontWeight: FontWeight.w600,
                          color: _hasDetectedClone
                              ? Colors.redAccent
                              : Colors.amberAccent,
                        ),
                      ),
                    ),
                    const Icon(Icons.chevron_right, size: 16, color: Colors.white54),
                  ],
                ),
              ),
            ),
          ],

          if (showStepUp) ...[
            _StepUpCard(
              level: _frame.level,
              onDismiss: () => setState(() => _stepUpDismissed = true),
            ),
            const SizedBox(height: 14),
          ],

          // Sticky AI Clone Warning Banner (persists across brief silences)
          if (showCloneBanner) ...[
            Container(
              margin: const EdgeInsets.symmetric(horizontal: 0, vertical: 8.0),
              padding:
                  const EdgeInsets.symmetric(horizontal: 14.0, vertical: 10.0),
              decoration: BoxDecoration(
                color: _frame.isAiClone
                    ? Colors.red.withValues(alpha: 0.22)
                    : Colors.red.withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(10.0),
                border: Border.all(
                  color: _frame.isAiClone ? Colors.redAccent : Colors.red.shade400,
                  width: 1.5,
                ),
              ),
              child: Row(
                children: [
                  Icon(
                    _frame.isAiClone
                        ? Icons.record_voice_over
                        : Icons.security_update_warning,
                    color: Colors.redAccent,
                    size: 26,
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Text(
                              _frame.isAiClone
                                  ? 'AI VOICE CLONE DETECTED'
                                  : 'STICKY THREAT LATCHED',
                              style: const TextStyle(
                                color: Colors.redAccent,
                                fontWeight: FontWeight.bold,
                                fontSize: 13,
                              ),
                            ),
                            if (!_frame.isAiClone) ...[
                              const SizedBox(width: 6),
                              Container(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 6, vertical: 1),
                                decoration: BoxDecoration(
                                  color: Colors.red.withValues(alpha: 0.3),
                                  borderRadius: BorderRadius.circular(4),
                                ),
                                child: const Text(
                                  'HOLD',
                                  style: TextStyle(
                                    fontSize: 9,
                                    fontWeight: FontWeight.bold,
                                    color: Colors.redAccent,
                                  ),
                                ),
                              ),
                            ],
                          ],
                        ),
                        const SizedBox(height: 2),
                        Text(
                          _frame.isAiClone
                              ? '${_frame.cloneConfidencePct}% confidence · Neural vocoder synthesis artifacts identified'
                              : 'AI clone previously flagged (Peak: $_peakCloneConfidence%) · Maintained across pause',
                          style: TextStyle(
                              color: Colors.red.shade200, fontSize: 11),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ],

          Center(child: RiskGauge(frame: _frame, size: 236)),
          const SizedBox(height: 10),
          Center(
            child: Text(
              connected
                  ? (_frame.reasons.isEmpty && _frame.risk == 0
                      ? 'Listening...'
                      : 'Live analysis')
                  : 'Not in a call',
              style: const TextStyle(fontSize: 12, color: Colors.white54),
            ),
          ),
          const SizedBox(height: 20),
          Row(
            children: [
              Expanded(
                child: FilledButton.icon(
                  onPressed: (connected || connecting) ? null : _answer,
                  icon: const Icon(Icons.phone_in_talk),
                  label: Text(connecting ? 'Joining...' : 'Answer'),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: FilledButton.icon(
                  style: FilledButton.styleFrom(backgroundColor: Colors.red),
                  onPressed: (connected || connecting) ? _hangUp : null,
                  icon: const Icon(Icons.call_end),
                  label: const Text('Hang up'),
                ),
              ),
            ],
          ),
          if (!connected && (_incidents.isNotEmpty || _hasDetectedClone)) ...[
            const SizedBox(height: 10),
            OutlinedButton.icon(
              onPressed: _showPostCallReport,
              icon: const Icon(Icons.receipt_long, size: 18),
              label: const Text('View Post-Call Forensic Report'),
            ),
          ],
          if (_state == CallState.failed && _rtc.lastError != null) ...[
            const SizedBox(height: 14),
            Text(
              _rtc.lastError!,
              style: const TextStyle(
                  fontSize: 12, color: Colors.redAccent, height: 1.35),
            ),
          ],
          const SizedBox(height: 24),
          if (_frame.reasons.isNotEmpty) ...[
            Text('Why', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            for (final r in _frame.reasons)
              Padding(
                padding: const EdgeInsets.only(bottom: 6),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Padding(
                      padding: EdgeInsets.only(top: 5, right: 8),
                      child: Icon(Icons.arrow_right, size: 16),
                    ),
                    Expanded(
                      child: Text(r,
                          style: const TextStyle(fontSize: 13, height: 1.35)),
                    ),
                  ],
                ),
              ),
            const SizedBox(height: 18),
          ],
          if (_frame.components.isNotEmpty) ...[
            Text('Signals', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            for (final e in _frame.components.entries)
              _ComponentBar(name: e.key, value: e.value),
            const SizedBox(height: 18),
          ],
          if (_frame.transcript.isNotEmpty) ...[
            Text('Heard', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 6),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.white10,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(_frame.transcript,
                  style: const TextStyle(fontSize: 13, height: 1.4)),
            ),
            const SizedBox(height: 6),
            const Text(
              'Transcription is context only. It never decides the verdict -- '
              'urgent words from a verified human are not an attack.',
              style: TextStyle(fontSize: 11, color: Colors.white54, height: 1.35),
            ),
            const SizedBox(height: 18),
          ],
          Text(
            [
              if (_frame.pSynthetic != null)
                'raw p(synthetic) ${_frame.pSynthetic!.toStringAsFixed(2)}',
              if (_frame.inferenceMs != null)
                'inference ${_frame.inferenceMs!.toStringAsFixed(0)} ms',
              if (_frame.droppedWindows > 0)
                'dropped ${_frame.droppedWindows}',
            ].join('   ·   '),
            style: const TextStyle(fontSize: 11, color: Colors.white38),
          ),
        ],
      ),
    );
  }
}

/// Comprehensive Post-Call Forensic Report dialog presented upon hang-up.
class _PostCallReportDialog extends StatelessWidget {
  final String sessionId;
  final DateTime? callStartTime;
  final DateTime? callEndTime;
  final List<ThreatIncident> incidents;
  final int peakCloneConfidence;
  final double lowestAuthenticity;
  final ThreatLevel peakThreatLevel;
  final bool hasDetectedClone;
  final List<String> cumulativeReasons;
  final VoidCallback onReset;

  const _PostCallReportDialog({
    required this.sessionId,
    required this.callStartTime,
    required this.callEndTime,
    required this.incidents,
    required this.peakCloneConfidence,
    required this.lowestAuthenticity,
    required this.peakThreatLevel,
    required this.hasDetectedClone,
    required this.cumulativeReasons,
    required this.onReset,
  });

  String _formatDuration() {
    if (callStartTime == null) return '0s';
    final end = callEndTime ?? DateTime.now();
    final diff = end.difference(callStartTime!);
    final m = diff.inMinutes;
    final s = diff.inSeconds % 60;
    return m > 0 ? '${m}m ${s}s' : '${s}s';
  }

  String _formatTime(DateTime dt) {
    return '${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}:${dt.second.toString().padLeft(2, '0')}';
  }

  void _copyToClipboard(BuildContext context) {
    final buffer = StringBuffer();
    buffer.writeln('=== DEFVOICE FORENSIC CALL REPORT ===');
    buffer.writeln('Session ID       : $sessionId');
    buffer.writeln('Call Duration    : ${_formatDuration()}');
    buffer.writeln('Overall Verdict  : ${hasDetectedClone ? "CRITICAL: AI VOICE CLONE" : (peakThreatLevel == ThreatLevel.suspicious ? "SUSPICIOUS CALL" : "AUTHENTIC")}');
    buffer.writeln('Peak Clone Conf  : $peakCloneConfidence%');
    buffer.writeln('Min Authenticity : ${lowestAuthenticity.toStringAsFixed(0)}%');
    buffer.writeln('Threat Incidents : ${incidents.length}');
    buffer.writeln('\n--- CUMULATIVE REASONS ---');
    if (cumulativeReasons.isEmpty) {
      buffer.writeln('No threat reasons flagged.');
    } else {
      for (final r in cumulativeReasons) {
        buffer.writeln('• $r');
      }
    }
    buffer.writeln('\n--- INCIDENT TIMELINE ---');
    if (incidents.isEmpty) {
      buffer.writeln('No incidents recorded.');
    } else {
      for (final inc in incidents) {
        buffer.writeln('[${_formatTime(inc.timestamp)}] Clone Conf: ${inc.cloneConfidencePct}%, Auth: ${inc.authenticityPct}%');
        for (final r in inc.reasons) {
          buffer.writeln('   - $r');
        }
      }
    }
    buffer.writeln('======================================');

    Clipboard.setData(ClipboardData(text: buffer.toString()));
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Forensic audit report copied to clipboard!'),
        duration: Duration(seconds: 2),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final isClone = hasDetectedClone || peakCloneConfidence >= 50;
    final isSuspicious = !isClone && peakThreatLevel == ThreatLevel.suspicious;

    final verdictColor = isClone
        ? Colors.redAccent
        : (isSuspicious ? Colors.orangeAccent : Colors.greenAccent);
    final verdictTitle = isClone
        ? 'CRITICAL THREAT: AI CLONE DETECTED'
        : (isSuspicious
            ? 'SUSPICIOUS CALL DETECTED'
            : 'AUTHENTIC CALL VERIFIED');
    final verdictSubtitle = isClone
        ? 'Neural vocoder synthesis artifacts & voiceprint divergence detected.'
        : (isSuspicious
            ? 'Urgent keyword patterns or elevated speaker variance noted.'
            : 'No synthetic acoustic signatures or scam indicators found.');

    return Dialog(
      backgroundColor: const Color(0xFF18181B),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      insetPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 24),
      child: ConstrainedBox(
        constraints: BoxConstraints(
          maxHeight: MediaQuery.of(context).size.height * 0.85,
          maxWidth: 500,
        ),
        child: Column(
          children: [
            // Modal Header
            Container(
              padding: const EdgeInsets.fromLTRB(20, 16, 12, 16),
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.05),
                borderRadius:
                    const BorderRadius.vertical(top: Radius.circular(16)),
              ),
              child: Row(
                children: [
                  Icon(Icons.fact_check_outlined, color: verdictColor, size: 24),
                  const SizedBox(width: 10),
                  const Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Post-Call Forensic Report',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            color: Colors.white,
                          ),
                        ),
                        Text(
                          'DefVoice Deepfake & Audio Analysis',
                          style: TextStyle(fontSize: 11, color: Colors.white54),
                        ),
                      ],
                    ),
                  ),
                  IconButton(
                    icon: const Icon(Icons.close, size: 20, color: Colors.white70),
                    onPressed: () => Navigator.of(context).pop(),
                  ),
                ],
              ),
            ),

            // Scrollable Content
            Expanded(
              child: ListView(
                padding: const EdgeInsets.all(20),
                children: [
                  // Verdict Banner
                  Container(
                    padding: const EdgeInsets.all(14),
                    decoration: BoxDecoration(
                      color: verdictColor.withValues(alpha: 0.15),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: verdictColor, width: 1.5),
                    ),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Icon(
                          isClone
                              ? Icons.gpp_bad_outlined
                              : (isSuspicious
                                  ? Icons.gpp_maybe_outlined
                                  : Icons.gpp_good_outlined),
                          color: verdictColor,
                          size: 28,
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                verdictTitle,
                                style: TextStyle(
                                  color: verdictColor,
                                  fontWeight: FontWeight.bold,
                                  fontSize: 13,
                                ),
                              ),
                              const SizedBox(height: 4),
                              Text(
                                verdictSubtitle,
                                style: TextStyle(
                                  color: Colors.white.withValues(alpha: 0.85),
                                  fontSize: 11.5,
                                  height: 1.35,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),

                  // Session & Stat Grid
                  Row(
                    children: [
                      _StatCard(
                        title: 'Peak Clone Conf',
                        value: '$peakCloneConfidence%',
                        color: peakCloneConfidence >= 50
                            ? Colors.redAccent
                            : Colors.white70,
                      ),
                      const SizedBox(width: 8),
                      _StatCard(
                        title: 'Min Authenticity',
                        value: '${lowestAuthenticity.toStringAsFixed(0)}%',
                        color: lowestAuthenticity < 40
                            ? Colors.redAccent
                            : (lowestAuthenticity < 70
                                ? Colors.orangeAccent
                                : Colors.greenAccent),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Row(
                    children: [
                      _StatCard(
                        title: 'Threat Incidents',
                        value: '${incidents.length}',
                        color: incidents.isNotEmpty
                            ? Colors.redAccent
                            : Colors.white70,
                      ),
                      const SizedBox(width: 8),
                      _StatCard(
                        title: 'Call Duration',
                        value: _formatDuration(),
                        color: Colors.white70,
                      ),
                    ],
                  ),
                  const SizedBox(height: 18),

                  // Cumulative Reasons
                  if (cumulativeReasons.isNotEmpty) ...[
                    const Text(
                      'Flagged Indicators',
                      style: TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                    const SizedBox(height: 8),
                    for (final r in cumulativeReasons)
                      Padding(
                        padding: const EdgeInsets.only(bottom: 6),
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Padding(
                              padding: EdgeInsets.only(top: 5, right: 8),
                              child: Icon(Icons.label_important_outline,
                                  size: 14, color: Colors.redAccent),
                            ),
                            Expanded(
                              child: Text(
                                r,
                                style: const TextStyle(
                                  fontSize: 12,
                                  color: Colors.white70,
                                  height: 1.35,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    const SizedBox(height: 18),
                  ],

                  // Incident Timeline Log
                  const Text(
                    'Incident Timeline',
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 8),
                  if (incidents.isEmpty)
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: Colors.white.withValues(alpha: 0.05),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: const Text(
                        'No anomalous synthesis or high-risk incidents detected during this session.',
                        style: TextStyle(fontSize: 11.5, color: Colors.white54),
                      ),
                    )
                  else
                    for (int i = 0; i < incidents.length; i++)
                      Container(
                        margin: const EdgeInsets.only(bottom: 8),
                        padding: const EdgeInsets.all(10),
                        decoration: BoxDecoration(
                          color: Colors.white.withValues(alpha: 0.04),
                          borderRadius: BorderRadius.circular(8),
                          border: Border.all(
                            color: Colors.white.withValues(alpha: 0.08),
                          ),
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              mainAxisAlignment:
                                  MainAxisAlignment.spaceBetween,
                              children: [
                                Text(
                                  _formatTime(incidents[i].timestamp),
                                  style: const TextStyle(
                                    fontSize: 11,
                                    color: Colors.white54,
                                    fontFamily: 'monospace',
                                  ),
                                ),
                                Container(
                                  padding: const EdgeInsets.symmetric(
                                      horizontal: 6, vertical: 2),
                                  decoration: BoxDecoration(
                                    color: Colors.red.withValues(alpha: 0.2),
                                    borderRadius: BorderRadius.circular(4),
                                  ),
                                  child: Text(
                                    '${incidents[i].cloneConfidencePct}% Clone',
                                    style: const TextStyle(
                                      fontSize: 10,
                                      fontWeight: FontWeight.bold,
                                      color: Colors.redAccent,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                            if (incidents[i].reasons.isNotEmpty) ...[
                              const SizedBox(height: 4),
                              Text(
                                incidents[i].reasons.join(' · '),
                                style: const TextStyle(
                                    fontSize: 11, color: Colors.white70),
                              ),
                            ],
                          ],
                        ),
                      ),
                  const SizedBox(height: 18),

                  // Mitigations Box
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.blueGrey.withValues(alpha: 0.15),
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(
                        color: Colors.blueGrey.withValues(alpha: 0.3),
                      ),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Row(
                          children: [
                            Icon(Icons.info_outline,
                                size: 16, color: Colors.lightBlueAccent),
                            SizedBox(width: 6),
                            Text(
                              'Forensic Directives',
                              style: TextStyle(
                                fontSize: 12,
                                fontWeight: FontWeight.bold,
                                color: Colors.lightBlueAccent,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 6),
                        Text(
                          isClone
                              ? '• Do NOT approve any requested transactions, OTP transfers, or wire requests.\n'
                                '• Perform immediate out-of-band verification via known stored phone number.\n'
                                '• Log this session timestamp ($sessionId) with your security / fraud team.'
                              : '• Call verified authentic within normal bounds.\n'
                                '• Remain cautious of standard social engineering tactics.',
                          style: const TextStyle(
                              fontSize: 11.5, color: Colors.white70, height: 1.4),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),

            // Modal Actions Footer
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.05),
                borderRadius:
                    const BorderRadius.vertical(bottom: Radius.circular(16)),
              ),
              child: Row(
                children: [
                  OutlinedButton.icon(
                    onPressed: () => _copyToClipboard(context),
                    icon: const Icon(Icons.copy, size: 15),
                    label: const Text('Copy Log', style: TextStyle(fontSize: 12)),
                  ),
                  const Spacer(),
                  TextButton(
                    onPressed: onReset,
                    child: const Text('Reset', style: TextStyle(fontSize: 12)),
                  ),
                  const SizedBox(width: 8),
                  FilledButton(
                    onPressed: () => Navigator.of(context).pop(),
                    child: const Text('Close', style: TextStyle(fontSize: 12)),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _StatCard extends StatelessWidget {
  final String title;
  final String value;
  final Color color;

  const _StatCard({
    required this.title,
    required this.value,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        decoration: BoxDecoration(
          color: Colors.white.withValues(alpha: 0.05),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              title,
              style: const TextStyle(fontSize: 10.5, color: Colors.white54),
            ),
            const SizedBox(height: 2),
            Text(
              value,
              style: TextStyle(
                fontSize: 15,
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Shown whenever the backend is running the stub countermeasure.
///
/// Do not remove this to make screenshots look better. A placeholder presented
/// as a detector is the one thing that turns a good project into a dishonest
/// one, and judges do ask what the model is.
class _DegradedBanner extends StatelessWidget {
  const _DegradedBanner();

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.amber.withValues(alpha: 0.14),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.amber),
      ),
      child: const Row(
        children: [
          Icon(Icons.science_outlined, size: 18, color: Colors.amber),
          SizedBox(width: 10),
          Expanded(
            child: Text(
              'DEGRADED: stub countermeasure. Scores are a placeholder '
              'heuristic, not a trained detector.',
              style: TextStyle(fontSize: 12, height: 1.35),
            ),
          ),
        ],
      ),
    );
  }
}

/// The actual product surface. Detection is worthless if it does not change what
/// the person on the call does next, so the recommendation is an out-of-band
/// verification step, not "hang up".
class _StepUpCard extends StatelessWidget {
  final ThreatLevel level;
  final VoidCallback onDismiss;

  const _StepUpCard({required this.level, required this.onDismiss});

  @override
  Widget build(BuildContext context) {
    final c = colorForLevel(level);
    return Container(
      padding: const EdgeInsets.all(15),
      decoration: BoxDecoration(
        color: c.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: c, width: 1.5),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.verified_user_outlined, color: c, size: 20),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  level == ThreatLevel.critical
                      ? 'Do not act on this call'
                      : 'Verify before acting',
                  style: TextStyle(
                      fontWeight: FontWeight.w700, color: c, fontSize: 15),
                ),
              ),
              IconButton(
                onPressed: onDismiss,
                icon: const Icon(Icons.close, size: 18),
                visualDensity: VisualDensity.compact,
              ),
            ],
          ),
          const SizedBox(height: 6),
          const Text(
            'This voice may be synthetic. Hang up and call the person back on '
            'their known number, or ask something only they could answer. Do '
            'not approve a payment or read out an OTP on this call.',
            style: TextStyle(fontSize: 12.5, height: 1.4),
          ),
        ],
      ),
    );
  }
}

class _ComponentBar extends StatelessWidget {
  final String name;
  final double value; // 0-100

  const _ComponentBar({required this.name, required this.value});

  static const _labels = {
    'cm': 'Synthetic-speech model',
    'asv': 'Speaker mismatch',
    'context': 'Urgency / payment language',
    'is_ai_clone': 'AI Clone Detected',
    'clone_confidence_pct': 'Clone confidence %',
  };

  @override
  Widget build(BuildContext context) {
    final frac = (value.clamp(0, 100)) / 100.0;
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(_labels[name] ?? name,
                    style: const TextStyle(fontSize: 12.5)),
              ),
              Text(value.toStringAsFixed(0),
                  style: const TextStyle(fontSize: 12, color: Colors.white54)),
            ],
          ),
          const SizedBox(height: 4),
          ClipRRect(
            borderRadius: BorderRadius.circular(3),
            child: LinearProgressIndicator(
              value: frac,
              minHeight: 6,
              backgroundColor: const Color(0xFF27272A),
            ),
          ),
        ],
      ),
    );
  }
}

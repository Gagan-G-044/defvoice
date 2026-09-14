import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../models/telemetry.dart';

const Color kSafe = Color(0xFF16A34A);
const Color kCaution = Color(0xFFF59E0B);
const Color kSuspicious = Color(0xFFEA580C);
const Color kCritical = Color(0xFFDC2626);

Color colorForLevel(ThreatLevel l) {
  switch (l) {
    case ThreatLevel.safe:
      return kSafe;
    case ThreatLevel.caution:
      return kCaution;
    case ThreatLevel.suspicious:
      return kSuspicious;
    case ThreatLevel.critical:
      return kCritical;
  }
}

/// Authenticity gauge.
///
/// Shows `100 - risk` because "87% authentic" is immediately legible while
/// "13% risk" makes people hunt for the scale. It is the same number, and the
/// level ring underneath is what actually carries the verdict.
///
/// The needle is animated over 400ms. That is cosmetic, not smoothing -- the
/// real hysteresis lives in the backend's risk aggregator, and adding a second
/// filter here would make the on-stage latency number a lie.
class RiskGauge extends StatelessWidget {
  final RiskFrame frame;
  final double size;

  const RiskGauge({super.key, required this.frame, this.size = 240});

  @override
  Widget build(BuildContext context) {
    final color = colorForLevel(frame.level);
    return SizedBox(
      width: size,
      height: size,
      child: TweenAnimationBuilder<double>(
        tween: Tween<double>(begin: 100, end: frame.authenticityPct),
        duration: const Duration(milliseconds: 400),
        curve: Curves.easeOut,
        builder: (context, value, _) {
          return CustomPaint(
            painter: _GaugePainter(value: value, color: color),
            child: Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    value.clamp(0, 100).toStringAsFixed(0),
                    style: TextStyle(
                      fontSize: size * 0.26,
                      fontWeight: FontWeight.w700,
                      color: color,
                      height: 1.0,
                    ),
                  ),
                  Text(
                    '% authentic',
                    style: TextStyle(
                      fontSize: size * 0.065,
                      color: Colors.white70,
                      letterSpacing: 0.5,
                    ),
                  ),
                  const SizedBox(height: 6),
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 10,
                      vertical: 3,
                    ),
                    decoration: BoxDecoration(
                      color: color.withValues(alpha: 0.18),
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(color: color, width: 1),
                    ),
                    child: Text(
                      threatLevelLabel(frame.level),
                      style: TextStyle(
                        fontSize: size * 0.055,
                        fontWeight: FontWeight.w700,
                        color: color,
                        letterSpacing: 1.0,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }
}

class _GaugePainter extends CustomPainter {
  final double value;   // authenticity, 0-100
  final Color color;

  // 270-degree sweep starting bottom-left, the usual dial convention.
  static const double _start = math.pi * 0.75;
  static const double _sweep = math.pi * 1.5;

  _GaugePainter({required this.value, required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final stroke = size.width * 0.075;
    final rect = Rect.fromCircle(
      center: Offset(size.width / 2, size.height / 2),
      radius: (size.width - stroke) / 2,
    );

    final track = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = stroke
      ..strokeCap = StrokeCap.round
      ..color = const Color(0xFF27272A);
    canvas.drawArc(rect, _start, _sweep, false, track);

    final frac = (value.clamp(0, 100)) / 100.0;
    final arc = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = stroke
      ..strokeCap = StrokeCap.round
      ..color = color;
    // Drawn from the high-authenticity end so the arc *shrinks* as risk rises.
    canvas.drawArc(rect, _start, _sweep * frac, false, arc);

    // Threshold ticks at the backend's decision boundaries, expressed in
    // authenticity: risk 30/55/75 -> 70/45/25. Having them visible means the
    // gauge can be read against the state machine instead of on faith.
    for (final t in const [70.0, 45.0, 25.0]) {
      final a = _start + _sweep * (t / 100.0);
      final inner = rect.width / 2 - stroke * 0.8;
      final outer = rect.width / 2 + stroke * 0.15;
      final c = rect.center;
      canvas.drawLine(
        Offset(c.dx + inner * math.cos(a), c.dy + inner * math.sin(a)),
        Offset(c.dx + outer * math.cos(a), c.dy + outer * math.sin(a)),
        Paint()
          ..color = const Color(0xFF52525B)
          ..strokeWidth = 1.5,
      );
    }
  }

  @override
  bool shouldRepaint(_GaugePainter old) =>
      old.value != value || old.color != color;
}

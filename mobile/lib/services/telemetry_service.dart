import 'dart:async';
import 'dart:convert';

import 'package:web_socket_channel/web_socket_channel.dart';

import '../config.dart';
import '../models/telemetry.dart';

/// Subscriber to `/ws/telemetry/{session}`.
///
/// Read-only: this socket never influences the call. It carries risk frames to
/// Phone B and to the judge dashboard, and it reconnects on its own, because a
/// dropped telemetry socket must degrade to a stale gauge rather than to a
/// dead-looking app mid-demo.
class TelemetryService {
  WebSocketChannel? _ws;
  StreamSubscription? _sub;
  Timer? _keepAlive;
  Timer? _reconnect;
  bool _closed = false;

  final _frames = StreamController<RiskFrame>.broadcast();
  Stream<RiskFrame> get frames => _frames.stream;

  final _status = StreamController<String>.broadcast();

  /// 'connecting' | 'live' | 'reconnecting'
  Stream<String> get status => _status.stream;

  /// Last value of `attack_mode` reported by Phone A. Shown as a label only; the
  /// verdict never uses it.
  String? attackMode;

  /// True once the backend has told us it is running the stub countermeasure.
  bool degraded = false;

  void connect() {
    _closed = false;
    _open();
  }

  void _open() {
    _status.add('connecting');
    try {
      _ws = WebSocketChannel.connect(Uri.parse(AppConfig.telemetryUrl()));
    } catch (e) {
      _scheduleReconnect();
      return;
    }

    _sub = _ws!.stream.listen(
      (raw) {
        _status.add('live');
        Map<String, dynamic> msg;
        try {
          msg = jsonDecode(raw as String) as Map<String, dynamic>;
        } catch (_) {
          return;
        }
        switch (msg['type']) {
          case 'risk':
            if (msg['degraded'] == true) degraded = true;
            _frames.add(RiskFrame.fromJson(msg));
            break;
          case 'hello':
            degraded = msg['degraded'] == true;
            break;
          case 'attack_mode':
            attackMode = msg['mode'] as String?;
            break;
        }
      },
      onError: (_) => _scheduleReconnect(),
      onDone: _scheduleReconnect,
      cancelOnError: false,
    );

    _keepAlive?.cancel();
    _keepAlive = Timer.periodic(const Duration(seconds: 10), (_) {
      try {
        _ws?.sink.add('ping');
      } catch (_) {}
    });
  }

  void _scheduleReconnect() {
    if (_closed) return;
    _status.add('reconnecting');
    _keepAlive?.cancel();
    _sub?.cancel();
    _sub = null;
    _ws = null;
    _reconnect?.cancel();
    _reconnect = Timer(const Duration(seconds: 2), () {
      if (!_closed) _open();
    });
  }

  Future<void> dispose() async {
    _closed = true;
    _keepAlive?.cancel();
    _reconnect?.cancel();
    await _sub?.cancel();
    try {
      await _ws?.sink.close();
    } catch (_) {}
    await _frames.close();
    await _status.close();
  }
}

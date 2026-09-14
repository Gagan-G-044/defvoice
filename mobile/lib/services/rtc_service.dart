import 'dart:async';
import 'dart:convert';

import 'package:flutter_webrtc/flutter_webrtc.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

import '../config.dart';

enum CallRole { caller, receiver }

enum CallState { idle, connecting, connected, failed, ended }

/// WebRTC peer connection to the laptop middlebox.
///
/// Both phones dial the *laptop*, not each other. The laptop relays audio
/// between them and taps a copy for analysis. Consequences worth knowing:
///
///  * No STUN or TURN. Everything is on one LAN, so host candidates are enough
///    and `iceServers` is deliberately empty.
///  * No trickle ICE. We wait for gathering to finish and send one complete SDP,
///    which removes an entire class of signaling race conditions.
///  * The laptop is in the media path. If the backend dies, the call dies.
class RtcService {
  RTCPeerConnection? _pc;
  MediaStream? _localStream;
  WebSocketChannel? _ws;
  StreamSubscription? _wsSub;

  final _stateCtrl = StreamController<CallState>.broadcast();
  Stream<CallState> get stateStream => _stateCtrl.stream;
  CallState state = CallState.idle;

  String? lastError;

  void _setState(CallState s) {
    state = s;
    if (!_stateCtrl.isClosed) _stateCtrl.add(s);
  }

  /// [rawCapture] disables the WebRTC audio front-end (echo cancellation, noise
  /// suppression, AGC, high-pass).
  ///
  /// Phone A must set this true. Its "attack mode" plays a cloned clip out of
  /// the loudspeaker and re-captures it on its own mic, because Android exposes
  /// no custom audio source to inject a file into an outgoing track. With AEC
  /// enabled, WebRTC recognises the loudspeaker signal as echo and cancels the
  /// attack into silence -- you get a live call with nothing in it.
  ///
  /// Phone B should leave this false; it is a normal handset.
  Future<void> connect({
    required CallRole role,
    bool rawCapture = false,
  }) async {
    _setState(CallState.connecting);
    lastError = null;
    try {
      _pc = await createPeerConnection({
        'iceServers': const [],
        'sdpSemantics': 'unified-plan',
      });

      _pc!.onConnectionState = (RTCPeerConnectionState s) {
        switch (s) {
          case RTCPeerConnectionState.RTCPeerConnectionStateConnected:
            _setState(CallState.connected);
            break;
          case RTCPeerConnectionState.RTCPeerConnectionStateFailed:
            lastError = 'ICE failed -- are both devices on the same subnet?';
            _setState(CallState.failed);
            break;
          case RTCPeerConnectionState.RTCPeerConnectionStateClosed:
          case RTCPeerConnectionState.RTCPeerConnectionStateDisconnected:
            _setState(CallState.ended);
            break;
          default:
            break;
        }
      };

      // Remote audio renders automatically on Android; route it to the speaker
      // so judges can hear the call.
      _pc!.onTrack = (RTCTrackEvent event) {
        if (event.track.kind == 'audio') {
          Helper.setSpeakerphoneOn(true);
        }
      };

      _localStream = await navigator.mediaDevices.getUserMedia(
        _audioConstraints(rawCapture),
      );
      for (final track in _localStream!.getAudioTracks()) {
        await _pc!.addTrack(track, _localStream!);
      }

      await _openSignaling(role);
    } catch (e) {
      lastError = e.toString();
      _setState(CallState.failed);
      rethrow;
    }
  }

  Map<String, dynamic> _audioConstraints(bool rawCapture) {
    final on = !rawCapture;
    return {
      'audio': {
        // Newer flutter_webrtc reads these directly...
        'echoCancellation': on,
        'noiseSuppression': on,
        'autoGainControl': on,
        // ...older Android builds only honour the goog* mandatory block.
        'mandatory': {
          'googEchoCancellation': on,
          'googEchoCancellation2': on,
          'googNoiseSuppression': on,
          'googNoiseSuppression2': on,
          'googAutoGainControl': on,
          'googHighpassFilter': on,
        },
        'optional': const [],
      },
      'video': false,
    };
  }

  // -------------------------------------------------------------------------
  Future<void> _openSignaling(CallRole role) async {
    final roleName = role == CallRole.caller ? 'caller' : 'receiver';
    _ws = WebSocketChannel.connect(Uri.parse(AppConfig.signalUrl(roleName)));

    final answered = Completer<void>();

    _wsSub = _ws!.stream.listen(
      (raw) {
        Map<String, dynamic> msg;
        try {
          msg = jsonDecode(raw as String) as Map<String, dynamic>;
        } catch (_) {
          return;
        }
        switch (msg['type']) {
          case 'answer':
            _pc
                ?.setRemoteDescription(
                  RTCSessionDescription(msg['sdp'] as String, 'answer'),
                )
                .then((_) {
                  if (!answered.isCompleted) answered.complete();
                })
                .catchError((e) {
                  if (!answered.isCompleted) answered.completeError(e);
                });
            // The backend tells us when it is running on the stub CM. Surfaced
            // in the UI so nobody mistakes a placeholder for a detector.
            degraded = msg['degraded'] == true;
            break;
          case 'error':
            lastError = msg['message'] as String? ?? 'backend error';
            _setState(CallState.failed);
            if (!answered.isCompleted) answered.completeError(lastError!);
            break;
        }
      },
      onError: (e) {
        lastError = 'signaling: $e';
        _setState(CallState.failed);
        if (!answered.isCompleted) answered.completeError(e);
      },
      onDone: () {
        if (state == CallState.connecting) {
          lastError = 'signaling closed before answer -- check token and host';
          _setState(CallState.failed);
        }
        if (!answered.isCompleted) {
          answered.completeError(StateError('signaling closed'));
        }
      },
      cancelOnError: false,
    );

    final offer = await _pc!.createOffer({
      'offerToReceiveAudio': true,
      'offerToReceiveVideo': false,
    });
    await _pc!.setLocalDescription(offer);
    await _waitForIceGathering();

    final local = await _pc!.getLocalDescription();
    _ws!.sink.add(jsonEncode({'type': 'offer', 'sdp': local!.sdp}));

    // The receiver's offer is held by the backend until the caller's track
    // exists, so this can legitimately take a while. Cap it well under the
    // backend's own 30s wait so we fail with a message instead of hanging.
    await answered.future.timeout(
      const Duration(seconds: 25),
      onTimeout: () {
        lastError = 'no answer from backend -- is the caller connected?';
        _setState(CallState.failed);
        throw TimeoutException('no answer');
      },
    );
  }

  /// True when the backend is running the stub countermeasure.
  bool degraded = false;

  /// Wait for host candidates instead of trickling them.
  ///
  /// On a LAN this finishes in tens of milliseconds. The timeout exists only so
  /// a stuck gatherer degrades into "try the SDP we have" rather than a hang.
  Future<void> _waitForIceGathering() async {
    if (_pc == null) return;
    final done = Completer<void>();
    _pc!.onIceGatheringState = (RTCIceGatheringState s) {
      if (s == RTCIceGatheringState.RTCIceGatheringStateComplete &&
          !done.isCompleted) {
        done.complete();
      }
    };
    // Cheap poll as a backstop: some platform builds fire the callback before
    // the handler is installed.
    Timer.periodic(const Duration(milliseconds: 120), (t) async {
      if (done.isCompleted) {
        t.cancel();
        return;
      }
      final desc = await _pc?.getLocalDescription();
      if (desc != null && desc.sdp != null && desc.sdp!.contains('candidate')) {
        t.cancel();
        if (!done.isCompleted) done.complete();
      }
    });
    await done.future.timeout(
      const Duration(seconds: 3),
      onTimeout: () {},
    );
  }

  /// Tell the backend which mode Phone A is in. Display only -- the verdict is
  /// computed from audio alone, or the demo would be rigged.
  void reportAttackMode(String mode) {
    try {
      _ws?.sink.add(jsonEncode({'type': 'attack_mode', 'mode': mode}));
    } catch (_) {
      // Signaling may already be gone; not worth failing a call over.
    }
  }

  Future<void> hangUp() async {
    try {
      _ws?.sink.add(jsonEncode({'type': 'bye'}));
    } catch (_) {}
    await _wsSub?.cancel();
    _wsSub = null;
    try {
      await _ws?.sink.close();
    } catch (_) {}
    _ws = null;

    for (final track in _localStream?.getTracks() ?? <MediaStreamTrack>[]) {
      try {
        await track.stop();
      } catch (_) {}
    }
    await _localStream?.dispose();
    _localStream = null;

    try {
      await _pc?.close();
    } catch (_) {}
    _pc = null;
    _setState(CallState.ended);
  }

  Future<void> dispose() async {
    await hangUp();
    await _stateCtrl.close();
  }
}

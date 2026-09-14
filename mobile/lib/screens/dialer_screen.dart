import 'dart:async';

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/material.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart' show Helper;

import '../config.dart';
import '../services/rtc_service.dart';

/// Phone A. Places the call and, optionally, runs the attack.
///
/// HOW THE ATTACK ACTUALLY WORKS
/// -----------------------------
/// `flutter_webrtc` has no custom audio source on Android: you cannot hand a WAV
/// file to an outgoing track. So the clip is played out of this phone's
/// loudspeaker and re-captured by its own microphone. That is why
/// [RtcService.connect] is called with `rawCapture: true` -- with AEC on, WebRTC
/// classifies the loudspeaker signal as echo and cancels the attack into
/// silence, and you get a live call with nothing in it.
///
/// The acoustic round-trip is not a cheat: it adds room reverb and speaker
/// colouration, which makes the clone *harder* to detect, not easier. If you
/// need a deterministic, reverb-free injection for a recorded run, use
/// `backend/tools/attacker_cli.py`, which streams the WAV straight into RTP as a
/// compromised SIP endpoint would.
class DialerScreen extends StatefulWidget {
  const DialerScreen({super.key});

  @override
  State<DialerScreen> createState() => _DialerScreenState();
}

class _DialerScreenState extends State<DialerScreen> {
  final _rtc = RtcService();
  final _player = AudioPlayer();
  StreamSubscription? _stateSub;
  StreamSubscription? _playerSub;

  CallState _state = CallState.idle;
  String? _playingLabel;
  String _mode = 'live_mic';

  @override
  void initState() {
    super.initState();
    _stateSub = _rtc.stateStream.listen((s) {
      if (mounted) setState(() => _state = s);
    });
    _playerSub = _player.onPlayerComplete.listen((_) {
      if (!mounted) return;
      setState(() {
        _playingLabel = null;
        _mode = 'live_mic';
      });
      _rtc.reportAttackMode('live_mic');
    });
    _configureAudio();
  }

  /// Route playback through the voice-communication stream and force the
  /// loudspeaker, so the clip lands in the mic rather than the earpiece.
  ///
  /// If a future audioplayers release renames these enums, the minimum viable
  /// version is `AudioContextAndroid(isSpeakerphoneOn: true)`.
  Future<void> _configureAudio() async {
    try {
      await AudioPlayer.global.setAudioContext(
        AudioContext(
          android: const AudioContextAndroid(
            isSpeakerphoneOn: true,
            stayAwake: true,
            contentType: AndroidContentType.speech,
            usageType: AndroidUsageType.voiceCommunication,
            // Do not take audio focus: grabbing it can duck or interrupt the
            // WebRTC stream we are trying to inject into.
            audioFocus: AndroidAudioFocus.none,
          ),
        ),
      );
      await _player.setVolume(1.0);
      await _player.setReleaseMode(ReleaseMode.stop);
    } catch (_) {
      // Non-fatal: playback still works, it may just come out of the earpiece.
    }
  }

  @override
  void dispose() {
    _stateSub?.cancel();
    _playerSub?.cancel();
    _player.dispose();
    _rtc.dispose();
    super.dispose();
  }

  // -------------------------------------------------------------------------
  Future<void> _connect() async {
    try {
      // rawCapture: true -- see the class docstring. Non-negotiable for Phone A.
      await _rtc.connect(role: CallRole.caller, rawCapture: true);
      await Helper.setSpeakerphoneOn(true);
      _rtc.reportAttackMode(_mode);
    } catch (e) {
      if (!mounted) return;
      _toast(_rtc.lastError ?? '$e');
    }
  }

  Future<void> _hangUp() async {
    await _player.stop();
    await _rtc.hangUp();
    if (mounted) setState(() => _playingLabel = null);
  }

  Future<void> _play(CloneClip clip) async {
    // audioplayers prepends 'assets/', so strip it from the configured path.
    final path = clip.asset.startsWith('assets/')
        ? clip.asset.substring('assets/'.length)
        : clip.asset;
    setState(() {
      _playingLabel = clip.label;
      _mode = 'clone:${clip.label}';
    });
    _rtc.reportAttackMode(_mode);
    try {
      await _player.stop();
      await _player.play(AssetSource(path), volume: 1.0);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _playingLabel = null;
        _mode = 'live_mic';
      });
      _toast('Could not play ${clip.label}. Is the WAV in assets/clones/? ($e)');
    }
  }

  Future<void> _stopPlayback() async {
    await _player.stop();
    if (!mounted) return;
    setState(() {
      _playingLabel = null;
      _mode = 'live_mic';
    });
    _rtc.reportAttackMode('live_mic');
  }

  void _toast(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(msg), duration: const Duration(seconds: 5)),
    );
  }

  // -------------------------------------------------------------------------
  @override
  Widget build(BuildContext context) {
    final connected = _state == CallState.connected;
    final connecting = _state == CallState.connecting;

    return Scaffold(
      appBar: AppBar(title: const Text('Phone A -- Caller')),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          _StatusBar(state: _state, error: _rtc.lastError),
          const SizedBox(height: 18),
          Row(
            children: [
              Expanded(
                child: FilledButton.icon(
                  onPressed: (connected || connecting) ? null : _connect,
                  icon: const Icon(Icons.call),
                  label: Text(connecting ? 'Connecting...' : 'Place call'),
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
          const SizedBox(height: 26),
          Text('Attack soundboard',
              style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 4),
          const Text(
            'Playing a clip sends it out of the loudspeaker; this phone\'s own '
            'mic picks it up and it goes down the wire. Hold the phone close to '
            'nothing in particular -- room noise is realistic.',
            style: TextStyle(fontSize: 12.5, color: Colors.white70, height: 1.35),
          ),
          const SizedBox(height: 14),
          for (final clip in AppConfig.cloneClips) ...[
            _ClipTile(
              clip: clip,
              playing: _playingLabel == clip.label,
              enabled: connected,
              onPlay: () => _play(clip),
              onStop: _stopPlayback,
            ),
            const SizedBox(height: 10),
          ],
          const SizedBox(height: 18),
          Container(
            padding: const EdgeInsets.all(13),
            decoration: BoxDecoration(
              color: Colors.white10,
              borderRadius: BorderRadius.circular(8),
            ),
            child: const Text(
              'The backend is told which mode this phone is in, but never uses '
              'it to score anything -- the verdict comes from the audio alone. '
              'Say that out loud during the demo; it is the first thing a good '
              'judge will suspect.',
              style: TextStyle(fontSize: 12, color: Colors.white70, height: 1.4),
            ),
          ),
        ],
      ),
    );
  }
}

class _StatusBar extends StatelessWidget {
  final CallState state;
  final String? error;

  const _StatusBar({required this.state, this.error});

  @override
  Widget build(BuildContext context) {
    late final Color c;
    late final String label;
    switch (state) {
      case CallState.connected:
        c = Colors.green;
        label = 'In call';
        break;
      case CallState.connecting:
        c = Colors.amber;
        label = 'Connecting to laptop...';
        break;
      case CallState.failed:
        c = Colors.red;
        label = 'Failed';
        break;
      case CallState.ended:
        c = Colors.white38;
        label = 'Call ended';
        break;
      case CallState.idle:
        c = Colors.white38;
        label = 'Idle';
        break;
    }
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: c.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: c),
      ),
      child: Row(
        children: [
          Icon(Icons.circle, size: 11, color: c),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(label,
                    style: const TextStyle(fontWeight: FontWeight.w600)),
                if (state == CallState.failed && error != null) ...[
                  const SizedBox(height: 4),
                  Text(error!,
                      style: const TextStyle(
                          fontSize: 12, color: Colors.white70, height: 1.35)),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _ClipTile extends StatelessWidget {
  final CloneClip clip;
  final bool playing;
  final bool enabled;
  final VoidCallback onPlay;
  final VoidCallback onStop;

  const _ClipTile({
    required this.clip,
    required this.playing,
    required this.enabled,
    required this.onPlay,
    required this.onStop,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: EdgeInsets.zero,
      color: playing ? const Color(0xFF3F1D1D) : null,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(14, 12, 8, 12),
        child: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(clip.label,
                      style: const TextStyle(fontWeight: FontWeight.w600)),
                  const SizedBox(height: 3),
                  Text(clip.transcript,
                      style: const TextStyle(
                          fontSize: 12, color: Colors.white60, height: 1.3)),
                ],
              ),
            ),
            IconButton(
              onPressed: enabled ? (playing ? onStop : onPlay) : null,
              icon: Icon(playing ? Icons.stop_circle : Icons.play_circle_fill),
              iconSize: 34,
              color: playing ? Colors.red : Colors.white,
            ),
          ],
        ),
      ),
    );
  }
}

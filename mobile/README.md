# DefVoice mobile client

Two roles in one APK. Install the same build on both phones and pick the role on
the first screen.

## This is not a complete Flutter project yet

Only `lib/`, `pubspec.yaml` and `android/app/src/main/AndroidManifest.xml` are
here. The Gradle wrapper, `MainActivity`, icons and iOS scaffolding are machine-
generated and would be noise in a repo. Generate them once:

```bash
cd mobile
flutter create --platforms=android --org com.sih26104 --project-name defvoice .
```

`flutter create` will not overwrite files that already exist, so `pubspec.yaml`,
`lib/` and the manifest survive. If it does clobber the manifest (older Flutter
versions do), restore it from git — the permissions and
`android:usesCleartextTraffic` in it are load-bearing.

Then:

```bash
flutter pub get
```

## Required Gradle edit

`flutter_webrtc` needs API 23+. Open `android/app/build.gradle.kts` (or
`build.gradle` on older templates) and set:

```kotlin
android {
    compileSdk = 35
    defaultConfig {
        minSdk = 23        // flutter_webrtc floor; the default 21 fails to build
        targetSdk = 34
    }
}
```

## Clone clips

Drop your generated WAVs into `assets/clones/` using the filenames in
`lib/config.dart`, or edit that list to match what you have. Mono 16 kHz or
48 kHz both work — playback resamples.

Keep the clips out of git. They are a cloned voice of a real consenting person,
which is exactly the kind of file that should not exist in a public repo.

## Install on both phones

```bash
flutter devices                       # confirm both serials are listed
flutter run -d <serial-A>
flutter run -d <serial-B>
```

Or build once and push:

```bash
flutter build apk --release
adb -s <serial> install -r build/app/outputs/flutter-apk/app-release.apk
```

## Before every demo

1. `ipconfig` / `ip addr` on the laptop, put that address in the Laptop IP field.
2. Tap **Test backend** on both phones. Do not proceed until both say the backend
   is up — a failure here is a network problem, and it is much cheaper to find
   now than during the call.
3. Phone A: **Place call** first, then Phone B: **Answer**. The backend holds
   Phone B's offer until Phone A's audio track exists, so the reverse order
   works but wastes 30 seconds if Phone A never connects.

If **Test backend** times out, the venue AP is almost certainly isolating
clients. Fall back to USB, which needs no working Wi-Fi at all:

```bash
adb -s <serial-A> reverse tcp:8000 tcp:8000
adb -s <serial-B> reverse tcp:8000 tcp:8000
```

then set Laptop IP to `127.0.0.1` on both phones.

## Two things about Phone A that look like bugs

**Echo cancellation is off.** Phone A requests raw capture, because the attack
works by playing a clip out of the loudspeaker and letting its own mic pick it
up — AEC would recognise that as echo and cancel the attack into silence. The
side effect is that when you *speak* live on Phone A, Phone B may hear echo. Use
a wired headset on Phone A for the genuine-call part of the demo, or keep the
phones in separate rooms.

**Clone playback comes out of the loudspeaker, loudly.** That is the injection
path, not a volume bug. If you need a clean, reverb-free, repeatable injection,
use `backend/tools/attacker_cli.py` instead; it streams the WAV straight into RTP
the way a compromised SIP endpoint would.

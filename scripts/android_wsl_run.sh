#!/usr/bin/env bash
set -euo pipefail

# WSL helper for the Android demo:
# - Optionally provision/start a Windows emulator
# - Build APK from WSL
# - Install + launch via Windows adb.exe

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANDROID_DIR="$ROOT_DIR/src/android"
ASSETS_DIR="$ANDROID_DIR/app/src/main/assets"
DEFAULT_APP_ID="com.example.accompaniment"
DEFAULT_ACTIVITY=".MainActivity"

AVD_NAME="${AVD_NAME:-Pixel35}"
API_LEVEL="${API_LEVEL:-35}"
IMAGE_FLAVOR="${IMAGE_FLAVOR:-google_apis}"
ARCH="${ARCH:-x86_64}"
DEVICE_PROFILE="${DEVICE_PROFILE:-pixel}"
SETUP_EMULATOR=0
START_EMULATOR=0
SKIP_BUILD=0

usage() {
  cat <<'EOF'
Usage: scripts/android_wsl_run.sh [options]

Options:
  --setup-emulator      Create SDK image + AVD on Windows if missing
  --start-emulator      Start emulator on Windows if no emulator device connected
  --avd <name>          AVD name (default: Pixel35)
  --api <level>         Android API level (default: 35)
  --skip-build          Skip Gradle build/install, only launch app
  -h, --help            Show this help

Environment overrides:
  APP_ID                Android package name (default: com.example.accompaniment)
  MAIN_ACTIVITY         Launcher activity (default: .MainActivity)
  WIN_ADB_WINPATH       Full Windows path to adb.exe
  WIN_SDK_ROOT_WINPATH  Full Windows path to Android SDK root
EOF
}

log() {
  echo "[android-wsl] $*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

to_wsl_path() {
  wslpath "$1" | tr -d '\r'
}

win_cmd_output() {
  powershell.exe -NoProfile -Command "$1" | tr -d '\r'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --setup-emulator)
      SETUP_EMULATOR=1
      shift
      ;;
    --start-emulator)
      START_EMULATOR=1
      shift
      ;;
    --avd)
      AVD_NAME="${2:?Missing value for --avd}"
      shift 2
      ;;
    --api)
      API_LEVEL="${2:?Missing value for --api}"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

require_cmd powershell.exe
require_cmd wslpath

APP_ID="${APP_ID:-$DEFAULT_APP_ID}"
MAIN_ACTIVITY="${MAIN_ACTIVITY:-$DEFAULT_ACTIVITY}"

# Resolve Windows SDK root.
WIN_SDK_ROOT_WINPATH="${WIN_SDK_ROOT_WINPATH:-}"
if [[ -z "$WIN_SDK_ROOT_WINPATH" ]]; then
  WIN_SDK_ROOT_WINPATH="$(win_cmd_output '$sdk = $env:LOCALAPPDATA + "\Android\Sdk"; if (Test-Path $sdk) { $sdk }')"
fi
if [[ -z "$WIN_SDK_ROOT_WINPATH" ]]; then
  echo "Could not detect Windows Android SDK root. Set WIN_SDK_ROOT_WINPATH." >&2
  exit 1
fi

WIN_PLATFORM_TOOLS_WINPATH="$WIN_SDK_ROOT_WINPATH\\platform-tools"
WIN_EMULATOR_WINPATH="$WIN_SDK_ROOT_WINPATH\\emulator\\emulator.exe"
WIN_AVDMANAGER_WINPATH="$WIN_SDK_ROOT_WINPATH\\cmdline-tools\\latest\\bin\\avdmanager.bat"
WIN_SDKMANAGER_WINPATH="$WIN_SDK_ROOT_WINPATH\\cmdline-tools\\latest\\bin\\sdkmanager.bat"

WIN_ADB_WINPATH="${WIN_ADB_WINPATH:-$WIN_PLATFORM_TOOLS_WINPATH\\adb.exe}"
WIN_ADB="$(to_wsl_path "$WIN_ADB_WINPATH")"
LINUX_SDK_ROOT="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-$HOME/Android/Sdk}}"

if [[ ! -x "$WIN_ADB" ]]; then
  echo "adb.exe not found at: $WIN_ADB_WINPATH" >&2
  exit 1
fi

if [[ ! -d "$LINUX_SDK_ROOT" ]]; then
  echo "Android SDK directory not found at: $LINUX_SDK_ROOT" >&2
  echo "Set ANDROID_SDK_ROOT (or ANDROID_HOME) to your Linux SDK path." >&2
  exit 1
fi

if [[ -x "$ANDROID_DIR/gradlew" ]]; then
  BUILD_CMD=("./gradlew" ":app:assembleDebug")
else
  require_cmd gradle
  BUILD_CMD=("gradle" ":app:assembleDebug")
fi

SYSTEM_IMAGE="system-images;android-${API_LEVEL};${IMAGE_FLAVOR};${ARCH}"

if [[ "$SETUP_EMULATOR" -eq 1 ]]; then
  log "Ensuring emulator system image exists on Windows: $SYSTEM_IMAGE"
  powershell.exe -NoProfile -Command "& \"$WIN_SDKMANAGER_WINPATH\" \"$SYSTEM_IMAGE\"" >/dev/null

  log "Ensuring AVD exists on Windows: $AVD_NAME"
  AVD_EXISTS="$(win_cmd_output "& \"$WIN_EMULATOR_WINPATH\" -list-avds | Where-Object { \$_ -eq \"$AVD_NAME\" }")"
  if [[ -z "$AVD_EXISTS" ]]; then
    powershell.exe -NoProfile -Command "echo no | & \"$WIN_AVDMANAGER_WINPATH\" create avd -n \"$AVD_NAME\" -k \"$SYSTEM_IMAGE\" -d \"$DEVICE_PROFILE\"" >/dev/null
    log "Created AVD: $AVD_NAME"
  else
    log "AVD already exists: $AVD_NAME"
  fi
fi

DEVICES="$("$WIN_ADB" devices)"
if ! grep -qE '^emulator-[0-9]+\s+device$' <<<"$DEVICES"; then
  if [[ "$START_EMULATOR" -eq 1 || "$SETUP_EMULATOR" -eq 1 ]]; then
    log "Starting emulator on Windows: $AVD_NAME"
    powershell.exe -NoProfile -Command "Start-Process -FilePath \"$WIN_EMULATOR_WINPATH\" -ArgumentList '-avd $AVD_NAME'" >/dev/null
    log "Waiting for emulator to become ready..."
    "$WIN_ADB" wait-for-device >/dev/null
  else
    echo "No running emulator found. Pass --start-emulator (or --setup-emulator)." >&2
    exit 1
  fi
fi

if [[ ! -f "$ASSETS_DIR/chord_model.tflite" ]]; then
  if [[ -f "$ROOT_DIR/exports/chord_model.tflite" ]]; then
    log "Copying exports/chord_model.tflite into Android assets"
    cp "$ROOT_DIR/exports/chord_model.tflite" "$ASSETS_DIR/chord_model.tflite"
  else
    echo "Missing model file: $ASSETS_DIR/chord_model.tflite" >&2
    echo "Export first: python -m accompaniment.cli export --checkpoint outputs/checkpoints/best.pt" >&2
    exit 1
  fi
fi

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  log "Building debug APK"
  (
    cd "$ANDROID_DIR"
    ANDROID_HOME="$LINUX_SDK_ROOT" ANDROID_SDK_ROOT="$LINUX_SDK_ROOT" "${BUILD_CMD[@]}"
  )
  log "Installing debug APK"
  "$WIN_ADB" install -r "$ANDROID_DIR/app/build/outputs/apk/debug/app-debug.apk" >/dev/null
fi

log "Launching app: $APP_ID/$MAIN_ACTIVITY"
"$WIN_ADB" shell am start -n "$APP_ID/$MAIN_ACTIVITY" >/dev/null

log "Done. Emulator devices:"
"$WIN_ADB" devices

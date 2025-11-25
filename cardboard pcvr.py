import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math
import argparse
import json
import socket
import time
from pathlib import Path

# Parse CLI args early so we can support a safe dry-run mode and config file
parser = argparse.ArgumentParser(description='Driver4VR full-hand tracking')
parser.add_argument('--dry-run', action='store_true', help='Run without controlling the mouse or using the camera')
parser.add_argument('--config', '-c', help='Path to JSON config file (created if missing)')
parser.add_argument('--write-config', action='store_true', help='Write default config to --config path and exit')
args = parser.parse_args()
dry_run = args.dry_run
config_path = Path(args.config) if args.config else Path.cwd() / 'hand_col_config.json'
write_config = args.write_config

# Default configuration (created if missing)
default_config = {
    "pinch_threshold": 35,
    "thumb_threshold": 0.03,
    "smoothing": 5,
    "no_system_mouse": False,
    "mode": "system",
    "events": {
        "target": "udp",
        "udp_host": "127.0.0.1",
        "udp_port": 5005,
        "log_path": "hand_events.log"
    }
}

# vJoy options (optional, requires vJoy driver + pyvjoy)
default_config.update({
    'use_vjoy': False,
    'vjoy_device_right': 1,
    'vjoy_device_left': 2,
    'invert_right_y': False,
    'invert_left_y': False
})

if not config_path.exists():
    try:
        config_path.write_text(json.dumps(default_config, indent=2))
        print(f'Default config written to {config_path}')
    except Exception:
        print('Could not write default config file; continuing with defaults in memory')

# If user asked to write config and exit
if write_config:
    print(f'Wrote default config to {config_path} (or confirmed it exists). Exiting.')
    raise SystemExit(0)

# Load config (fall back to defaults on error)
config = default_config
try:
    config = json.loads(config_path.read_text())
except Exception:
    print('Failed to load config, using defaults')

# Apply config values
pinch_threshold = config.get('pinch_threshold', default_config['pinch_threshold'])
# thumb_threshold is relative (normalized hand coord space); keep small float
thumb_threshold = config.get('thumb_threshold', default_config['thumb_threshold'])

smoothing = config.get('smoothing', default_config['smoothing'])
no_system_mouse = config.get('no_system_mouse', default_config['no_system_mouse'])
mode = config.get('mode', default_config['mode'])
events_cfg = config.get('events', default_config['events'])

# vJoy config
use_vjoy = config.get('use_vjoy', default_config['use_vjoy'])
vjoy_device_right = config.get('vjoy_device_right', default_config['vjoy_device_right'])
vjoy_device_left = config.get('vjoy_device_left', default_config['vjoy_device_left'])
invert_right_y = config.get('invert_right_y', default_config['invert_right_y'])
invert_left_y = config.get('invert_left_y', default_config['invert_left_y'])

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Optional vJoy support: map each hand to a virtual joystick device so Driver4VR
# can see each hand as its own input. Requires vJoy driver + pyvjoy package.
# vJoy support removed per user request. The script no longer creates virtual
# joystick devices; it controls the system mouse and uses clicks/scrolls only.

# use smoothing from config
smooth = smoothing
prev_x, prev_y = 0, 0
enabled = True  # toggle gesture controls with backtick (`)

# If running in dry-run mode, monkeypatch pyautogui functions to safe no-ops
if dry_run:
    def _noop(*a, **k):
        print('DRY RUN pyautogui call:', a, k)

    try:
        pyautogui.moveTo = _noop
        pyautogui.mouseDown = lambda *a, **k: print('DRY RUN mouseDown', a, k)
        pyautogui.mouseUp = lambda *a, **k: print('DRY RUN mouseUp', a, k)
        pyautogui.hscroll = lambda *a, **k: print('DRY RUN hscroll', a, k)
        pyautogui.scroll = lambda *a, **k: print('DRY RUN scroll', a, k)
        pyautogui.size = lambda: (1280, 720)
    except Exception:
        # If pyautogui isn't present for any reason, continue — dry-run avoids control.
        pass

# Screen size (delayed until after possible dry-run monkeypatch)
screen_w, screen_h = pyautogui.size()

def dist(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


# Prepare event sink when mode includes 'events'
event_socket = None
event_log_path = None
if mode in ('events', 'both') and not dry_run:
    if events_cfg.get('target') == 'udp':
        try:
            event_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            event_socket.setblocking(False)
            print(f"Event UDP sender configured to {events_cfg.get('udp_host')}:{events_cfg.get('udp_port')}")
        except Exception as e:
            print('Failed to create event UDP socket:', e)
            event_socket = None
    elif events_cfg.get('target') == 'file':
        try:
            event_log_path = Path(events_cfg.get('log_path'))
            event_log_path.write_text('')
            print(f"Event log file configured at {event_log_path}")
        except Exception as e:
            print('Failed to open event log file:', e)
            event_log_path = None


def send_event(obj):
    obj['ts'] = time.time()
    payload = json.dumps(obj)
    if dry_run:
        print('EVENT:', payload)
        return
    if mode in ('events', 'both'):
        if events_cfg.get('target') == 'udp' and event_socket:
            try:
                event_socket.sendto(payload.encode('utf-8'), (events_cfg.get('udp_host'), int(events_cfg.get('udp_port'))))
            except Exception:
                pass
        if events_cfg.get('target') == 'file' and event_log_path:
            try:
                with open(event_log_path, 'a') as f:
                    f.write(payload + '\n')
            except Exception:
                pass

# vJoy setup (only if requested)
vjoy1 = vjoy2 = None
if use_vjoy and not dry_run:
    try:
        import pyvjoy
        try:
            vjoy1 = pyvjoy.VJoyDevice(vjoy_device_right)
            vjoy2 = pyvjoy.VJoyDevice(vjoy_device_left)
            print(f'vJoy devices {vjoy_device_right} and {vjoy_device_left} opened')
        except Exception as e:
            print('Failed to open vJoy devices:', e)
            vjoy1 = vjoy2 = None
    except Exception:
        print('pyvjoy not available; install pyvjoy and vJoy driver to enable virtual devices')
        vjoy1 = vjoy2 = None

cap = None
if not dry_run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam (index 0). Check camera connection.")

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    try:
        while True:
            if dry_run:
                # Synthetic blank frame for testing
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from camera; exiting loop.")
                    break

                frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                # Iterate over detected hands along with handedness labels
                # result.multi_handedness provides left/right labels corresponding
                # to multi_hand_landmarks ordering.
                for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                    lm = hand_landmarks.landmark
                    label = handedness.classification[0].label  # 'Left' or 'Right'

                    # Key landmarks
                    wrist = (int(lm[0].x * w), int(lm[0].y * h))
                    thumb_tip = (int(lm[4].x * w), int(lm[4].y * h))
                    thumb_base = (int(lm[2].x * w), int(lm[2].y * h))
                    index_tip = (int(lm[8].x * w), int(lm[8].y * h))

                    # Sensitivity / thresholds
                    # Compute pinch threshold in pixels: config may be pixels (>1) or normalized (<=1).
                    pinch_px = pinch_threshold if pinch_threshold > 1 else int(pinch_threshold * min(w, h))

                    # RIGHT hand: control cursor + left-click trigger + right-click grip
                    if label == 'Right':
                        mouse_x = np.interp(lm[0].x, [0, 1], [0, screen_w])
                        mouse_y = np.interp(lm[0].y, [0, 1], [0, screen_h])

                        smooth_x = prev_x + (mouse_x - prev_x) / smooth
                        smooth_y = prev_y + (mouse_y - prev_y) / smooth

                        if enabled and not no_system_mouse:
                            pyautogui.moveTo(smooth_x, smooth_y)
                        prev_x, prev_y = smooth_x, smooth_y

                        # Emit movement event (normalized + screen coords)
                        if mode in ('events', 'both'):
                            send_event({
                                'hand': 'Right',
                                'type': 'move',
                                'x': float(lm[0].x),
                                'y': float(lm[0].y),
                                'screen_x': float(smooth_x),
                                'screen_y': float(smooth_y)
                            })

                        # vJoy support removed: no virtual axis mapping

                        # Trigger (pinch) -> left mouse (use config pinch_px)
                        pinch = dist(index_tip, thumb_tip) < pinch_px
                        if enabled:
                            if pinch:
                                pyautogui.mouseDown()   # left down
                            else:
                                pyautogui.mouseUp()

                        # Emit pinch event
                        if mode in ('events', 'both'):
                            send_event({'hand': 'Right', 'type': 'pinch', 'state': bool(pinch)})

                        # no virtual button mapping (vJoy removed)

                        # Grip (fist) -> right mouse
                        fingers_open = (lm[8].y < lm[6].y and lm[12].y < lm[10].y)
                        grip = not fingers_open
                        if enabled:
                            if grip:
                                pyautogui.mouseDown(button='right')
                            else:
                                pyautogui.mouseUp(button='right')

                        # Emit grip event
                        if mode in ('events', 'both'):
                            send_event({'hand': 'Right', 'type': 'grip', 'state': bool(grip)})

                        # no virtual button mapping (vJoy removed)

                    # LEFT hand: joystick (thumb movement) + middle-click on pinch
                    else:  # label == 'Left'
                        # Thumb movement (use normalized landmark coordinates for thresholds)
                        thumb_dx_norm = lm[4].x - lm[2].x
                        thumb_dy_norm = lm[4].y - lm[2].y

                        if enabled:
                            # Horizontal joystick using normalized threshold
                            if thumb_dx_norm > thumb_threshold:
                                pyautogui.hscroll(-10)
                            elif thumb_dx_norm < -thumb_threshold:
                                pyautogui.hscroll(10)

                            # Vertical joystick
                            if thumb_dy_norm > thumb_threshold:
                                pyautogui.scroll(-10)
                            elif thumb_dy_norm < -thumb_threshold:
                                pyautogui.scroll(10)

                            # Left-hand pinch -> middle mouse (use pinch_px)
                            pinch_left = dist(index_tip, thumb_tip) < pinch_px
                            if pinch_left:
                                pyautogui.mouseDown(button='middle')
                            else:
                                pyautogui.mouseUp(button='middle')

                        # Emit joystick and pinch events for left hand
                        if mode in ('events', 'both'):
                            send_event({
                                'hand': 'Left',
                                'type': 'joystick',
                                'dx': float(thumb_dx_norm),
                                'dy': float(thumb_dy_norm)
                            })
                            send_event({'hand': 'Left', 'type': 'pinch', 'state': bool(pinch_left)})

                        # vJoy support removed: no virtual axis/button mapping for left hand

                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Status overlay
                status_text = "ENABLED" if enabled else "DISABLED"
                color = (0, 255, 0) if enabled else (0, 0, 255)
                cv2.putText(frame, f"Gesture Control: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Driver4VR Full-Hand Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to quit
                break
            elif key == ord('`'):
                # Toggle gesture control on/off. When turning off, ensure buttons are released.
                enabled = not enabled
                if not enabled:
                    try:
                        pyautogui.mouseUp()
                    except Exception:
                        pass
                    try:
                        pyautogui.mouseUp(button='right')
                    except Exception:
                        pass
                    try:
                        pyautogui.mouseUp(button='middle')
                    except Exception:
                        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
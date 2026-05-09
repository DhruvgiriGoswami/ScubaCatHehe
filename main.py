
# Scuba Cat Meme Detector
# -----------------------------------
# When you:
#   - keep ONE hand near your face
#   - move the OTHER hand side-to-side
#
# the scuba cat GIF appears 😭


import cv2
import mediapipe as mp
import numpy as np
import imageio
import time
from collections import deque

import pygame #for sound

# ----------------------------
# CONFIG
# ----------------------------

GIF_PATH = "scubaCatGif.gif"  #you can download a new gif and put the path here

HAND_FACE_DISTANCE = 0.25
WAVE_THRESHOLD = 0.025
WAVE_HISTORY = 12

SHOW_DEBUG = False #you can set this to true if you want to see your skeleton in the frames

# ----------------------------
# LOAD GIF
# ----------------------------

gif_frames = []
gif_index = 0

try:
    gif = imageio.mimread(GIF_PATH)

    for frame in gif:
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        gif_frames.append(frame)

    print(f"[+] Loaded GIF with {len(gif_frames)} frames")

except Exception as e:
    print("[!] Could not load scuba_cat.gif")
    print(e)
    print("[!] App will still run without GIF")


# ----------------------------
# AUDIO
# ----------------------------

pygame.mixer.init()

sound_loaded = False

try:
    pygame.mixer.music.load("scubaSound.mp3")
    sound_loaded = True
    print("[+] Loaded scuba.mp3")

except Exception as e:
    print("[!] Could not load scuba.mp3")
    print(e)


# ----------------------------
# MEDIAPIPE
# ----------------------------

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

# ----------------------------
# WEBCAM
# ----------------------------

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

wave_history = deque(maxlen=WAVE_HISTORY)

triggered = False
last_trigger = 0

# ----------------------------
# HELPERS
# ----------------------------

def distance(a, b):
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

def overlay_image(background, overlay, x, y, scale=0.4):
    h, w = overlay.shape[:2]

    new_w = int(w * scale)
    new_h = int(h * scale)

    overlay = cv2.resize(overlay, (new_w, new_h))

    bh, bw = background.shape[:2]

    if x + new_w > bw:
        new_w = bw - x

    if y + new_h > bh:
        new_h = bh - y

    if new_w <= 0 or new_h <= 0:
        return background

    overlay = overlay[:new_h, :new_w]

    roi = background[y:y+new_h, x:x+new_w]

    blended = cv2.addWeighted(roi, 0.3, overlay, 0.7, 0)

    background[y:y+new_h, x:x+new_w] = blended

    return background

# ----------------------------
# MAIN LOOP
# ----------------------------

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    small = cv2.resize(frame, (320, 240))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    h, w, _ = frame.shape

    triggered = False

    if results.pose_landmarks:

        lm = results.pose_landmarks.landmark

        nose = lm[mp_pose.PoseLandmark.NOSE]

        left_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

        # ----------------------------
        # HAND NEAR FACE
        # ----------------------------

        left_near_face = distance(left_wrist, nose) < HAND_FACE_DISTANCE
        right_near_face = distance(right_wrist, nose) < HAND_FACE_DISTANCE

        # ----------------------------
        # DETECT WAVING
        # ----------------------------

        # Use the hand NOT near the face for waving

        wave_hand_x = None

        if left_near_face:
            wave_hand_x = right_wrist.x

        elif right_near_face:
            wave_hand_x = left_wrist.x

        waving = False

        if wave_hand_x is not None:

            wave_history.append(float(wave_hand_x))

            if len(wave_history) >= WAVE_HISTORY:

                movement = np.std(wave_history)

                if movement > WAVE_THRESHOLD * 0.5:
                    waving = True

        # ----------------------------
        # FINAL TRIGGER
        # ----------------------------

        active_pose = (
            (left_near_face and waving)
            or
            (right_near_face and waving)
        )

        if active_pose:

            last_trigger = time.time()

            if sound_loaded and not pygame.mixer.music.get_busy():
                pygame.mixer.music.play(-1)

        else:

            if sound_loaded and pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
        # ----------------------------
        # DEBUG DRAW
        # ----------------------------

        if SHOW_DEBUG:
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            cv2.putText(
                frame,
                f"LEFT FACE: {left_near_face}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

            cv2.putText(
                frame,
                f"WAVING: {waving}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,0,0),
                2
            )

    # ----------------------------
    # KEEP EFFECT LIVE
    # ----------------------------

    if active_pose:

        cv2.putText(
            frame,
            "SCUBA SCUBA",
            (w//2 - 180, 80),
            cv2.FONT_HERSHEY_DUPLEX,
            1.5,
            (0,255,255),
            4
        )

        if gif_frames:
            gif_frame = gif_frames[gif_index % len(gif_frames)]
            gif_index += 1

            frame = overlay_image(
                frame,
                gif_frame,
                w//2 - 150,
                h//2 - 150,
                scale=0.8
            )

    # ----------------------------
    # SHOW WINDOW
    # ----------------------------

    cv2.imshow("Scuba Cat Detector", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

# ----------------------------
# CLEANUP
# ----------------------------

cap.release()
cv2.destroyAllWindows()
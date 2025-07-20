import numpy as np
import cv2

ORANGE = (0, 100, 255)
BLUE = (180, 220, 220)
PINK = (200, 180, 255)
VINTAGE = (80, 110, 150)


def mesh_face(frame, face_landmarks):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # Lấy toàn bộ các điểm landmark trên mặt
    points = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
              for lm in face_landmarks.landmark]

    # Tạo đường bao quanh khuôn mặt
    hull = cv2.convexHull(np.array(points))
    cv2.fillConvexPoly(mask, hull, 255)

    return mask

def Bilateral_filter(frame, mask):
    smoothed = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

    face_smoothed = cv2.bitwise_and(smoothed, smoothed, mask=mask)
    background = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))

    res = cv2.add(background, face_smoothed)
    return res

def blend_image(original, overlay_color, alpha=0.3):
    overlay = np.full_like(original, overlay_color, dtype=np.uint8)
    return cv2.addWeighted(overlay, alpha, original, 1 - alpha, 0)

def apply_filter(frame, filter_name):
    filters = {
        'none': lambda img: img,
        'warm_orange': lambda img: blend_image(img, ORANGE, 0.2),
        'cool_blue': lambda img: blend_image(img, BLUE, 0.25),
        'while_pink': lambda img: blend_image(img, PINK, 0.3),
        'vintage': lambda img: blend_image(img, VINTAGE, 0.25),
    }
    return filters.get(filter_name, filters['none'])(frame)


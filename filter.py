import numpy as np
import cv2

ORANGE = (0, 100, 255)
BLUE = (180, 220, 220)
PINK = (200, 180, 255)
VINTAGE = (80, 110, 150)

def apply_filter(frame, filter_idx):

    if filter_idx % 6 == 0:
        return frame  # Không filter

    elif filter_idx % 6 == 1:  # Sepia
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        sepia = cv2.transform(frame, sepia_filter)
        sepia = np.clip(sepia, 0, 255)
        return sepia.astype(np.uint8)

    elif filter_idx % 6 == 2:  # Grayscale
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    elif filter_idx % 6 == 3:  # Negative
        return cv2.bitwise_not(frame)

    elif filter_idx % 6 == 4:  # Brighten
        return cv2.convertScaleAbs(frame, alpha=1.2, beta=30)

    elif filter_idx % 6 == 5:  # Cartoon
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(blur, 255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 10)
        color = cv2.bilateralFilter(frame, d=9, sigmaColor=200, sigmaSpace=200)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon

    return frame
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


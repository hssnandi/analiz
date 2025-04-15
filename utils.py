import cv2

def draw_overlay(image):
    """
    رسم دایره‌ی راهنمای صورت روی تصویر.
    """
    h, w, _ = image.shape
    center_x = w // 2
    center_y = h // 2
    radius = min(h, w) // 4

    # رسم دایره خاکستری (پیش‌فرض)
    overlay = image.copy()
    cv2.circle(overlay, (center_x, center_y), radius, (128, 128, 128), 3)
    return overlay

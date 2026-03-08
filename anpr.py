import cv2
import numpy as np
import pytesseract

# ==============================
# TESSERACT PATH (WINDOWS)
# ==============================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ==============================
# START CAMERA
# ==============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera not accessible")
    exit()

print("Camera started. Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ==============================
    # IMAGE PREPROCESSING
    # ==============================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blur, 30, 200)

    # ==============================
    # FIND CONTOURS
    # ==============================
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    plate_contour = None

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            plate_contour = approx
            break

    plate_text = ""

    if plate_contour is not None:
        # DRAW RECTANGLE
        cv2.drawContours(frame, [plate_contour], -1, (0, 255, 0), 3)

        # ==============================
        # CROP PLATE
        # ==============================
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [plate_contour], 0, 255, -1)

        x, y = np.where(mask == 255)
        topx, topy = np.min(x), np.min(y)
        bottomx, bottomy = np.max(x), np.max(y)

        cropped_plate = gray[topx:bottomx+1, topy:bottomy+1]

        # ==============================
        # OCR
        # ==============================
        config = "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        plate_text = pytesseract.image_to_string(cropped_plate, config=config)
        plate_text = plate_text.strip()

        # DISPLAY TEXT
        cv2.putText(
            frame,
            plate_text,
            (topy, topx - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # ==============================
    # SHOW CAMERA
    # ==============================
    cv2.imshow("LIVE ANPR", frame)

    # PRESS Q TO QUIT
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# CLEANUP
# ==============================
cap.release()
cv2.destroyAllWindows()

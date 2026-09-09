import cv2
import time

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: DroidCam could not be opened")
    raise SystemExit(1)

print("DroidCam opened successfully")
print(
    "Resolution:",
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    "x",
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
)

start = time.perf_counter()
frames = 0

print("Running 30-second camera test...")
print("Press Q to stop early.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Frame read failed")
        break

    frames += 1

    cv2.imshow("DroidCam FPS Test - Press Q", frame)

    elapsed = time.perf_counter() - start

    if elapsed >= 30:
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

elapsed = time.perf_counter() - start

print()
print("========== RESULT ==========")
print("Frames:", frames)
print("Duration:", round(elapsed, 2), "seconds")
print("Average input FPS:", round(frames / elapsed, 2))
print("============================")
import cv2
from ultralytics import YOLO

model = YOLO("model/checkpoints/best.pt")
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model(frame, verbose=False)
    annotated = results[0].plot()

    cv2.imshow("Smoke test - press q to quit", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

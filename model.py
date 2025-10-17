import cv2
from ultralytics import YOLO

# Load your custom-trained YOLOv8 model
# The path points to the weights file from your specific training run.
model = YOLO('best.pt')

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam 

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")


# Loop through the video frames
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if ret:
        # Perform object detection on the frame using your custom model
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow('Custom Object Detection', annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
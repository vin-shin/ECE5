import cv2

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the USB camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

print("Press 'q' to quit.")

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the frame to grayscale (Haar cascade works on grayscale images)
    resized_frame = cv2.resize(frame, (320, 240))  # Adjust resolution as needed
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces and print coordinates
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Print the coordinates of the face
        print(f"Face detected at X: {x}, Y: {y}, Width: {w}, Height: {h}")

    # Display the frame with rectangles
    cv2.imshow('Face Detection', frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
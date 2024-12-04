import cv2
import serial

# Initialize UART communication
ser = serial.Serial('/dev/serial0', 9600, timeout=1)  # Adjust port and baud rate if necessary
ser.flush()

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the USB camera
cap = cv2.VideoCapture(0)

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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

    # Resize the frame for faster processing
    resized_frame = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the resized frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Get the first detected face's center
        x, y, w, h = faces[0]
        face_center_x = x + w // 2

        # Determine position relative to the frame
        frame_center_x = resized_frame.shape[1] // 2
        tolerance = 30  # Adjust tolerance as needed

        if face_center_x < frame_center_x - tolerance:
            command = "left\n"  # Face is to the left
            print("Sending: left")
        elif face_center_x > frame_center_x + tolerance:
            command = "right\n"  # Face is to the right
            print("Sending: right")
        else:
            command = "forward\n"  # Face is centered
            print("Sending: forward")

        # Send command to Arduino via UART
        ser.write(command.encode('utf-8'))

    # Display the frame with rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('Face Detection', frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
ser.close()
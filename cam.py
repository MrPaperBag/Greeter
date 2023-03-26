import cv2

# Load the pre-trained Haar Cascade classifier for detecting human faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")

if __name__ == "__main__":
    # Initialize the camera
    cap = cv2.VideoCapture(0)

    # Loop to capture and process each frame from the camera
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect human faces in the grayscale image using the Haar Cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        bodies = body_cascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (50, 100), # Min size for valid detection, changes according to video size or body size in the video.
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        # If one or more faces are detected, print a message that a human is present
        if len(faces) > 0 or len(bodies) > 0:
            print('Human detected!')


            print(f"{faces=}")
            print(f"{bodies=}")
        # Display the processed frame with any detected faces outlined
        for (x, y, w, h) in list(faces)+list(bodies):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('frame', frame)

        # Check if the user has pressed the 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the display window
    cap.release()
    cv2.destroyAllWindows()


def FindHumans(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect human faces in the grayscale image using the Haar Cascade classifier
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    bodies = body_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (50, 100), # Min size for valid detection, changes according to video size or body size in the video.
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    # If one or more faces are detected, print a message that a human is present
    if len(faces) > 0 or len(bodies) > 0:
        print('Human detected!')
        return True
    return False


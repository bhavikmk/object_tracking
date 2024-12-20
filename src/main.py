import cv2

def main():
    # Open webcam
    cap = cv2.VideoCapture('video.mp4')

    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    while True:
        ret, frame = cap.read()  # Capture a frame from the webcam
        if not ret:
            print("Error: Unable to read from the camera.")
            break

        # Resize the frame to 300 pixels width while maintaining aspect ratio
        height, width, _ = frame.shape
        new_width = 300
        new_height = int((new_width / width) * height)
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Display the resized frame
        cv2.imshow("Resized Webcam Feed (300px Width)", resized_frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

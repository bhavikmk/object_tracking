import cv2
import numpy as np

def main():
    # Open webcam
    cap = cv2.VideoCapture('video.mp4')
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Random colors for the tracks
    color = np.random.randint(0, 255, (100, 3))

    # Take the first frame and detect corners
    ret, old_frame = cap.read()
    if not ret:
        print("Error: Unable to read from the camera.")
        cap.release()
        return

    old_frame = cv2.resize(old_frame, (300, int(300 * old_frame.shape[0] / old_frame.shape[1])))  # Resize to 300px width
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from the camera.")
            break

        # Resize the frame
        frame = cv2.resize(frame, (300, int(300 * frame.shape[0] / frame.shape[1])))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        if p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Select good points
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # Draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                    frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

                frame = cv2.add(frame, mask)

                # Update the previous frame and points
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)

        # Display the frame
        cv2.imshow("Feature Tracking", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

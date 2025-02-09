import os
import time
import cv2

FILE_NAME_LIGHT_CLEAN_25 = "op25_light_clean.mp4"
FILE_NAME_LIGHT_CLEAN_50 = "op50_light_clean.mp4"
FILE_NAME_LIGHT_CLEAN_75 = "op75_light_clean.mp4"
FILE_NAME_LIGHT_CLEAN_100 = "op100_light_clean.mp4"

FILE_NAME_DARK_CLEAN_25 = "op25_dark_clean.mp4"
FILE_NAME_DARK_CLEAN_50 = "op50_dark_clean.mp4"
FILE_NAME_DARK_CLEAN_75 = "op75_dark_clean.mp4"
FILE_NAME_DARK_CLEAN_100 = "op100_dark_clean.mp4"

FILE_NAME_LIGHT_NOISY_25 = "op25_light_noisy.mp4"
FILE_NAME_LIGHT_NOISY_50 = "op50_light_noisy.mp4"
FILE_NAME_LIGHT_NOISY_75 = "op75_light_noisy.mp4"
FILE_NAME_LIGHT_NOISY_100 = "op100_light_noisy.mp4"

FILE_NAME_DARK_NOISY_25 = "op25_dark_noisy.mp4"
FILE_NAME_DARK_NOISY_50 = "op50_dark_noisy.mp4"
FILE_NAME_DARK_NOISY_75 = "op75_dark_noisy.mp4"
FILE_NAME_DARK_NOISY_100 = "op100_dark_noisy.mp4"


def main(fname=FILE_NAME_LIGHT_NOISY_25):
    cap = cv2.VideoCapture(0)  
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    video_dims = (int(cap.get(3)), int(cap.get(4)))
    out = cv2.VideoWriter(os.path.join("videos", fname), fourcc, 30.0, video_dims)
	
    # Check if camera is operational
    if not cap.isOpened():
        print("ERROR: Camera not opened!")
        quit()
	
    # loop runs if capturing has been initialized. 
    start_time = time.time()
    while(time.time() - start_time < 30):
        ret, frame = cap.read() 
        if not ret:
            print("ERROR: failed to grab frame at {} seconds".format(time.time() - start_time))
            break
        
        out.write(frame) 
        cv2.imshow('Original', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()  
    cv2.destroyAllWindows()
    print("Video saved to results.mp4")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()

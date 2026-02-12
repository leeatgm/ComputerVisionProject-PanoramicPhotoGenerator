import cv2
import numpy as np
import os
from PostProcess import post_process

# select and save key frames based on SIFT matching and RANSAC
# inspired from Lab 4 (Homography and Image Stitching)
def extract_frames(video_path, output_dir, stride=40,
                       min_inliers=100, max_inliers=500, ratio_thresh=0.6):

    # make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # open video file
    cap = cv2.VideoCapture(video_path)

    # initialize SIFT feature detector
    sift = cv2.SIFT_create()

    # read first frame and save as initial key frame
    ret, prev_frame = cap.read()

    key_frames = []     # store captured frames
    frame_idx = 0
    key_idx = 0

    # save first key frame
    key_frames.append(prev_frame)
    cv2.imwrite(os.path.join(output_dir, f"frame{key_idx}.jpg"), prev_frame)
    print(f"Captured key frame frame{key_idx}.jpg")
    key_idx += 1

    # compute width of region of interest (2/3 of full width)
    h, w = prev_frame.shape[:2]
    roi_w = w * 2 // 3

    # iterate through video frames
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # only process every stride-th frame
        if frame_idx % stride != 0:
            continue

        # convert ROIs to grayscale for feature detection
        prev_gray = cv2.cvtColor(prev_frame[:, -roi_w:], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame[:, :roi_w], cv2.COLOR_BGR2GRAY)

        # detect keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(prev_gray, None)
        kp2, des2 = sift.detectAndCompute(curr_gray, None)
        if des1 is None or des2 is None:
            continue

        # match descriptors using knn
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)

        # apply ratio test to filter good matches
        good = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
        if len(good) < 4:
            continue

        # prepare point arrays for homography
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        # compute homography with RANSAC to remove outliers
        _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        inliers = int(mask.sum())

        if not (min_inliers < inliers < max_inliers) :
            print(f"Skipping frame with inliers: {inliers}")    # show rejected frames inliers' value
            prev_frame = curr_frame                             # assign rejected stride to prev_frame
            continue
        else:
            # save frame only if inlier count is within range
            key_frames.append(curr_frame)
            cv2.imwrite(os.path.join(output_dir, f"frame{key_idx}.jpg"), curr_frame)
            print(f"Captured key frame frame{key_idx}.jpg (inliers: {inliers})")
            key_idx += 1                # move to next frame
            prev_frame = curr_frame     # move current frame to prev_frame


    cap.release()
    return key_frames

# stitch provided frames into a panorama
def stitch_images(images):
    stitcher = cv2.Stitcher_create()        # create stitcher object
    status, pano = stitcher.stitch(images)
    if status == cv2.STITCHER_OK:
        return pano
    else:
        print(f"Stitching failed: {status}")    # fail case
        return None

def main():
    video_path = 'canteen360.mp4'                         # 360 videos
    keyframe_dir = 'saved_frames'
    raw_panorama_path = 'panorama.jpg'
    processed_panorama_path = 'panorama_processed.jpg'

    # extract the frame
    print("Extracting key frames...")
    frames = extract_frames(video_path, keyframe_dir)

    # stitch the frame together with black border
    print("Stitching frames into panorama...")
    panorama = stitch_images(frames)
    if panorama is not None:
        cv2.imwrite(raw_panorama_path, panorama)
        print(f"Panorama saved as {raw_panorama_path}")
        print("Post-processing image now...")
        post_process(panorama, processed_panorama_path) # remove the border and blend the image
    else:
        print("Panorama generation failed.")

if __name__ == '__main__':
    main()

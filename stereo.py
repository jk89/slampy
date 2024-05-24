import cv2
import numpy as np
import time 
def get_orb_keypoint_and_descriptors(orb, frame):
    kp = orb.detect(frame,None)
    kp, des = orb.compute(frame, kp)
    return kp, des

def get_matcher():
    return cv2.BFMatcher(cv2.NORM_HAMMING) #, crossCheck=True

def get_covisible_features(matcher, des_frame1, des_frame2):
    return matcher.knnMatch(des_frame1,des_frame2, k=2)

def get_frame(cam):
    ret, frame = cam.read()
    return frame

def main():
    cam1 = cv2.VideoCapture(0)
    cam2 = cv2.VideoCapture(2)


    n_features = 1000
    factor = 1.2
    orb = cv2.ORB_create(n_features, factor)
    bf_matcher = get_matcher()
    prev_time = time.time()

    while True:
        frame1 = get_frame(cam1)
        #cv2.imshow("test0", frame1)
        frame2 = get_frame(cam2)
        #cv2.imshow("test1", frame2)
        
        kp_frame_1, des_frame_1 = get_orb_keypoint_and_descriptors(orb, frame1)
        kp_frame_2, des_frame_2 = get_orb_keypoint_and_descriptors(orb, frame2)
        
        try:
            matches = get_covisible_features(bf_matcher, des_frame_1, des_frame_2)
        except Exception as e:
            print(e)
            continue
        
        good_points = []
                
        # Lowe's test
        for x in matches:
            if len(x) > 1:
                m,n = x
                if m.distance < 0.75*n.distance:
                    good_points.append([m])
        
        # Extract pixel coordinates of matched keypoints
        src_pts = np.float32([kp_frame_1[m[0].queryIdx].pt for m in good_points]).reshape(-1, 2)
        dst_pts = np.float32([kp_frame_2[m[0].trainIdx].pt for m in good_points]).reshape(-1, 2)

        # Estimate fundamental matrix using RANSAC
        F, mask_fundamental = cv2.findFundamentalMat(src_pts, dst_pts, method=cv2.FM_RANSAC, ransacReprojThreshold=3.0, confidence=0.99, maxIters=2000)
        
        final_img = cv2.drawMatchesKnn(frame1,kp_frame_1,frame2,kp_frame_2,good_points,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        current_time = time.time()
        elapsed_time = current_time - prev_time
        fps = 1.0 / elapsed_time
        prev_time = current_time

        # Display FPS on the frame
        cv2.putText(final_img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Separate points into inliers and outliers based on the RANSAC mask
        if mask_fundamental is None:
            cv2.imshow("stereo", final_img) 
            continue
        
        inlier_mask = mask_fundamental.ravel() == 1
        outlier_mask = ~inlier_mask
                
        # Draw lines connecting inliers on both images
        for m in np.array(good_points)[inlier_mask]:
            pt1 = tuple(map(int, kp_frame_1[m[0].queryIdx].pt))
            pt2 = tuple(map(int, kp_frame_2[m[0].trainIdx].pt))
            final_img = cv2.line(final_img, pt1, (pt2[0] + frame1.shape[1], pt2[1]), (0, 255, 0), 2)

        # Draw lines connecting outliers on both images
        for m in np.array(good_points)[outlier_mask]:
            pt1 = tuple(map(int, kp_frame_1[m[0].queryIdx].pt))
            pt2 = tuple(map(int, kp_frame_2[m[0].trainIdx].pt))
            final_img = cv2.line(final_img, pt1, (pt2[0] + frame1.shape[1], pt2[1]), (0, 0, 255), 2)
            
        final_good_points = np.array(good_points)[inlier_mask]
        print("final_good_points", len(final_good_points))

        
        cv2.imshow("stereo", final_img)    
        k = cv2.waitKey(1)
        if k%256 == 27: # Escape
            break
    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
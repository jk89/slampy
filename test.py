import numpy as np
import cv2
import open3d as o3d

def visualize_3d_points(vis, points_3d, R, t):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    # Add the point cloud geometry
    vis.clear_geometries()
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    # Set the intrinsic parameters
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width=1920, height=1080, fx=525, fy=525, cx=1920/2, cy=1080/2)

    # Set the extrinsic parameters using R and t
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t.flatten()

    # Set camera parameters
    camera_params = o3d.camera.PinholeCameraParameters()
    camera_params.intrinsic = intrinsic
    camera_params.extrinsic = extrinsic

    # Convert and set camera parameters
    vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params)

    # Allow user interaction with the camera
    vis.run()

    # Get the updated camera parameters
    updated_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    updated_extrinsic = updated_params.extrinsic

def get_camera():
    cam = cv2.VideoCapture(0)
    return cam

def get_camera_frame(cam):
    return cam.read() # ret, frame

def get_orb_keypoint_and_descriptors(orb, frame):
    kp = orb.detect(frame,None)
    kp, des = orb.compute(frame, kp)
    return kp, des

def get_matcher():
    return cv2.BFMatcher(cv2.NORM_HAMMING) #, crossCheck=True

def get_flan_matcher():
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6, # 12
                    key_size = 12,     # 20
                    multi_probe_level = 1) #2
    search_params = dict(checks=50)   # or pass empty dictionary
    return cv2.FlannBasedMatcher(index_params,search_params)

def get_covisible_features(matcher, des_frame1, des_frame2):
    return matcher.knnMatch(des_frame1,des_frame2, k=2)

"""
DMatch

float 	distance
 
int 	imgIdx
 	train image index More...
 
int 	queryIdx
 	query descriptor index More...
 
int 	trainIdx
 	train descriptor index More...
  
"""


"""
cv2.KeyPoint

float 	angle
 
int 	class_id
 	object class (if the keypoints need to be clustered by an object they belong to) More...
 
int 	octave
 	octave (pyramid layer) from which the keypoint has been extracted More...
 
Point2f 	pt (has member x and y)
 	coordinates of the keypoints More...
 
float 	response
 	the response by which the most strong keypoints have been selected. Can be used for the further sorting or subsampling More...
 
float 	size
 	diameter of the meaningful keypoint neighborhood More...
  
"""

def main():
    camera = get_camera()
    n_features = 1000
    factor = 2.0
    orb = cv2.ORB_create(n_features, factor)
    bf_matcher = get_matcher()
    frames_and_kpdes = []
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    while True:
        ret, frame = get_camera_frame(camera)
        kp, des = get_orb_keypoint_and_descriptors(orb, frame)
        
        #print(kp) # (n) each a cv2.KeyPoint
        #print(des) # (n,32) this is the tensor of all orb features as a stack
        
        if len(frames_and_kpdes) > 1:
            last_frame_data = frames_and_kpdes[-1]
            last_frame = last_frame_data["frame"]
            last_frame_kp = last_frame_data["kp"]
            last_frame_des = last_frame_data["des"]
            
            current_frame = frame
            current_frame_kp = kp
            current_frame_des = des
            
            try:
                #print(des)
                #print(last_frame_des)
                matches = get_covisible_features(bf_matcher, des, last_frame_data["des"])
            except Exception as e:
                print(e)
                continue

            #print(matches)
            good_points = []
            
            for x in matches:
                if len(x) > 1:
                    m,n = x
                    if m.distance < 0.75*n.distance:
                        good_points.append([m])
                        
            #print("good points", good_points)
            #for point in good_points:
            #    print("point", point[0])
            
            #for match in good_points:
            #    print("match", len(match))
            #    queryKpIdx = match.queryIdx
            #    trainKpDescIdx = match.trainIdx
            #    queryPoint = current_frame_kp[queryKpIdx]
            #    trainPoint = last_frame_kp[trainKpDescIdx]
            #    print((queryPoint.pt[0], queryPoint.pt[1]), (trainPoint.pt[0], trainPoint.pt[1]))
            #    
            #    cv2.circle(current_frame, [int(current_frame_kp[queryKpIdx].pt[0]), int(current_frame_kp[queryKpIdx].pt[1])], int(queryPoint.size), [0,0,255], 1)    
            #current_frame[int(queryPoint[0]):int(queryPoint[1])] = [0,0,255]

            # Extract pixel coordinates of matched keypoints
            src_pts = np.float32([current_frame_kp[match[0].queryIdx].pt for match in good_points]).reshape(-1, 1, 2)
            dst_pts = np.float32([last_frame_kp[match[0].trainIdx].pt for match in good_points]).reshape(-1, 1, 2)

            # Estimate fundamental matrix using RANSAC
            F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)

            # Decompose fundamental matrix to obtain relative pose
            try:
                retval, R, t, mask = cv2.recoverPose(F, src_pts, dst_pts)
            except:
                continue
            
            print("R and T", R, t)

            # Create projection matrices
            K = np.eye(3)  # assuming identity intrinsic matrix for simplicity
            P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
            P2 = np.hstack((R, t))

            # Triangulate points
            points_4d = cv2.triangulatePoints(P1, P2, src_pts.reshape(-1, 2).T, dst_pts.reshape(-1, 2).T)

            # Convert homogeneous coordinates to 3D
            points_3d = cv2.convertPointsFromHomogeneous(points_4d.T).reshape(-1, 3)

            # Optional: Print or use 3D points as needed
            print("3D Points:", len(points_3d))

            # Optional: Visualize 3D points in the current frame
            #for point in points_3d:
            #   x, y, z = point
            #   cv2.circle(current_frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            visualize_3d_points(vis, points_3d, R, t)
            final_img = cv2.drawMatchesKnn(current_frame,current_frame_kp,last_frame,last_frame_kp,good_points,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            #final_img = cv2.drawMatches(current_frame, current_frame_kp,  last_frame, last_frame_kp, matches,None) # query first param then train
            cv2.imshow("test", final_img)
            
            
        if des is not None:
            frames_and_kpdes.append({"frame":frame, "kp": kp, "des": des})

        
        k = cv2.waitKey(1)
        if k%256 == 27: # Escape
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cv2.namedWindow("test")
    main()
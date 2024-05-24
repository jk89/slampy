from calib import detect_camera_indicies, load_camera_calibration
import cv2
import numpy as np
import time 
import open3d as o3d
from typing import Type, List, Sequence, TypedDict

# Define a StepType
class StepType(TypedDict):
    frame1_img: cv2.typing.MatLike
    frame2_img: cv2.typing.MatLike
    frame_1_kp: Sequence[cv2.KeyPoint]
    frame_2_kp: Sequence[cv2.KeyPoint]
    frame_1_kp_des: cv2.typing.MatLike
    frame_2_kp_des: cv2.typing.MatLike
    frame_1_pts_2d: np.ndarray[np.float32]
    frame_2_pts_2d: np.ndarray[np.float32]
    frame_1_R: cv2.typing.MatLike
    frame_1_T: cv2.typing.MatLike
    frame_2_R: cv2.typing.MatLike
    frame_2_T: cv2.typing.MatLike
    frame_1_P: cv2.typing.MatLike
    frame_2_P: cv2.typing.MatLike
    pts_3d: cv2.typing.MatLike
    
def visualize_all3(calibration, camera_indicies, steps: List[StepType]):
    # Step 0 - Init
    WIDTH = 1280
    HEIGHT = 720

    # Step 1 - Get scene origin sphere
    sphere_origin = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
    sphere_origin.translate([0.0, 0.0, 0.0])

    # Step 2 - Create visualizer object
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.create_window(width=WIDTH, height=HEIGHT)

    # Step 3 - Add objects to visualizer
    #visualizer.add_geometry(sphere_origin)

    # Combine all 3D points from each step
    all_points = np.concatenate([step["pts_3d"] for step in steps])

    # Visualize all 3D points
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(all_points)
    visualizer.add_geometry(point_cloud, reset_bounding_box=True)

    for step in steps:
        
        # Step 4 - Get camera lines
        intrinsic_matrix = calibration[camera_indicies[0]]["camera_matrix"]
        standard_camera_parameters_obj = visualizer.get_view_control().convert_to_pinhole_camera_parameters()
        
        # Extract rotation matrix ("R") and translation vector ("T") from the dictionary
        
        R = step["frame_1_R"]
        T = step["frame_1_T"]
        P = step["frame_1_P"]

        # Create a 4x4 extrinsic matrix
        #print("building custom matrix for extinsics", R, T, T.flatten())
        custom_extrinsic_matrix = np.eye(4)
        custom_extrinsic_matrix[:3, :3] = R
        custom_extrinsic_matrix[:3, 3] = T.flatten()
    
        #print("standard_camera_parameters_obj.extrinsic", standard_camera_parameters_obj.extrinsic)
        #print("custom_extrinsic_matrix", custom_extrinsic_matrix)
        camera_lines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=WIDTH, view_height_px=HEIGHT, intrinsic=intrinsic_matrix, extrinsic=custom_extrinsic_matrix)

        #visualizer.add_geometry(camera_lines)

        sphere_cam_1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere_cam_1.paint_uniform_color([1, 0, 0])  # Set the color to red
        sphere_cam_1.translate(P[:3, 3])  # Translate the sphere to the camera position
        visualizer.add_geometry(sphere_cam_1, reset_bounding_box=True)

        # Create an arrow representing the camera orientation
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.2, cone_height=0.1
        )
        
        arrow.transform(custom_extrinsic_matrix)  # Apply the camera's extrinsic transformation
        visualizer.add_geometry(arrow, reset_bounding_box=True)
        
        #### next

        # Step 4 - Get camera lines
        intrinsic_matrix = calibration[camera_indicies[1]]["camera_matrix"]
        
        # Extract rotation matrix ("R") and translation vector ("T") from the dictionary
        R = step["frame_2_R"]
        T = step["frame_2_T"]
        P = step["frame_2_P"]

        # Create a 4x4 extrinsic matrix
        #print("building custom matrix for extinsics", R, T, T.flatten())
        custom_extrinsic_matrix = np.eye(4)
        custom_extrinsic_matrix[:3, :3] = R
        custom_extrinsic_matrix[:3, 3] = T.flatten()
    
        #print("standard_camera_parameters_obj.extrinsic", standard_camera_parameters_obj.extrinsic)
        #print("custom_extrinsic_matrix", custom_extrinsic_matrix)
        camera_lines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=WIDTH, view_height_px=HEIGHT, intrinsic=intrinsic_matrix, extrinsic=custom_extrinsic_matrix)

        #visualizer.add_geometry(camera_lines)

        sphere_cam_2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere_cam_2.paint_uniform_color([0, 1, 0])  # Set the color to blue
        sphere_cam_2.translate(P[:3, 3])  # Translate the sphere to the camera position
        visualizer.add_geometry(sphere_cam_2, reset_bounding_box=True)
        
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.2, cone_height=0.1
        )
        arrow.transform(custom_extrinsic_matrix)  # Apply the camera's extrinsic transformation
        visualizer.add_geometry(arrow, reset_bounding_box=True)

    # Step 5 - Run visualizer
    visualizer.run()

def get_orb_keypoint_and_descriptors(orb:Type[cv2.ORB], frame):
    kp: Sequence[cv2.KeyPoint] = orb.detect(frame,None)
    kp_and_des: tuple[Sequence[cv2.KeyPoint], cv2.typing.MatLike] = orb.compute(frame, kp)
    kp = kp_and_des[0]
    des = kp_and_des[1]
    return kp, des

def get_matcher():
    return cv2.BFMatcher(cv2.NORM_HAMMING) #, crossCheck=True

def get_covisible_features(matcher:Type[cv2.BFMatcher], des_frame1, des_frame2):
    return matcher.knnMatch(des_frame1,des_frame2, k=2)

def get_frame(cam: cv2.VideoCapture):
    ret, frame = cam.read()
    return frame

def rectify_image(frame, config):
    h, w = frame.shape[:2]
    camera_matrix =  config["camera_matrix"]
    distortion_coeffs  = config["distortion_coeffs"]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, (w, h), 1, (w, h))
    rectified_frame = cv2.undistort(frame, camera_matrix, distortion_coeffs, None, new_camera_matrix)
    return rectified_frame

def triangulate_points(K1, R1, T1, K2, R2, T2, src_pts, dest_pts):
    # Convert keypoints to homogeneous coordinates

    # Projection matrices for both cameras
    P1 = np.dot(K1, np.hstack((R1, T1)))
    P2 = np.dot(K2, np.hstack((R2, T2)))

    # Triangulate points
    points_4d_homogeneous = cv2.triangulatePoints(P1, P2, src_pts, dest_pts)

    # Convert homogeneous coordinates to 3D
    points_3d = cv2.convertPointsFromHomogeneous(points_4d_homogeneous.T).reshape(-1, 3)

    return points_3d

def get_stereo_2d_correspondance(bf_matcher:Type[cv2.BFMatcher], frame_1_kp, frame_2_kp, frame_1_kp_des, frame_2_kp_des):
    try:
        matches = get_covisible_features(bf_matcher, frame_1_kp_des, frame_2_kp_des)
    except Exception as e:
        return (None,None,None)
        
    good_matches:Sequence[Sequence[cv2.DMatch]] = []
                
    # Lowe's test
    for x in matches:
        if len(x) > 1:
            m,n = x
            if m.distance < 0.75*n.distance:
                good_matches.append([m])
        
    # Extract pixel coordinates of matched keypoints
    frame_1_pts_2d :np.ndarray[(Ellipsis, 2), np.float32] = np.float32([frame_1_kp[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 2)
    frame_2_pts_2d:np.ndarray[(Ellipsis, 2), np.float32] = np.float32([frame_2_kp[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 2)

    return (good_matches, frame_1_pts_2d, frame_2_pts_2d)

def transform_points_to_origin(pts_3d, R1, T1):
    # Ensure pts_3d is a numpy array
    pts_3d = np.array(pts_3d)

    # Inverse of the rotation matrix
    inv_R1 = np.linalg.inv(R1)
    
    # Inverse of the translation vector
    inv_T1 = -T1
    
    # Homogeneous coordinates for 3D points
    pts_3d_homogeneous = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))

    # Transform the points to the origin reference frame
    transformation_matrix = np.concatenate((inv_R1.T, inv_T1), axis=1)
    transformed_pts = np.dot(transformation_matrix, pts_3d_homogeneous.T).T
    
    # Extract the first three columns as the final 3D points
    pts_3d_at_origin = transformed_pts[:, :3]

    return pts_3d_at_origin
"""
Objective:

Implement Pipeline

Initial map construction
Step one (initialise):
- Take a stereo pair of images
- Work out feature correspondances
- Use to estimate 2nd camera pose
- Define camera 1 pose as origin.
- Triangulate initial world points
- Collect features,stereo-matches,inliers-mask,map-points into the current step
Step two (track)
- Take a stereo pair of images
- For each stereo pair match their features to the previous frame (either same cam, or the alternate cam, or both)
- For the feature matches to the prior frames, find the inliers, connect these inliers to the prior frames map points
- SolvePnpRansac to recover camera pose estimates.
- RefinePnp to recover refined camera pose estimates.
- Triangulate new world points

Desired data structures
Map
Frame
Features


so we want for each step
frame1, frame2, inliers:{ frame_1_kp, frame_1_kp, kp_des1, kp_des2, map_points}, R1, T1, P1, R2, T2, P2



"""
    
def filter_stereo_correspondences_with_ransac(
    frame_1_kp: Sequence[cv2.KeyPoint],
    frame_2_kp: Sequence[cv2.KeyPoint],
    frame_1_kp_des: cv2.typing.MatLike,
    frame_2_kp_des: cv2.typing.MatLike,
    frame_1_pts_2d: np.ndarray[np.float32],
    frame_2_pts_2d: np.ndarray[np.float32],
    good_matches: list[Sequence[cv2.DMatch]],
    ransac_reproj_threshold: float = 3.0,
    confidence: float = 0.999,
    max_iters: int = 2000):
    
    ret = obtain_fundamental_and_inliers_with_ransac_2d(frame_1_pts_2d, frame_2_pts_2d, ransac_reproj_threshold, confidence, max_iters)
    F, mask_fundamental, inlier_mask = ret
    if mask_fundamental is None:
        return None, None, None, None, None, None, None, None, None, None, None
    
    # Separate points into inliers and outliers based on the RANSAC mask
    inlier_mask: Type[np.ndarray[bool]] = mask_fundamental.ravel() == 1
    outlier_mask: Type[np.ndarray[bool]] = ~inlier_mask
    inlier_matches: np.ndarray[cv2.DMatch] = np.array(good_matches)[inlier_mask]
    outlier_matches: np.ndarray[cv2.DMatch] = np.array(good_matches)[outlier_mask]
    
    inliers_frame_1_kp = []
    inliers_frame_2_kp = []
    inliers_frame_1_kp_des = []
    inliers_frame_2_kp_des = []
    inlier_frame_1_pts_2d: np.ndarray[np.float32] = frame_1_pts_2d[inlier_mask]
    inlier_frame_2_pts_2d: np.ndarray[np.float32] = frame_2_pts_2d[inlier_mask]
    
    #print("inlier_matches", len(inlier_matches))
    #print("frame_1_pts_2d", inlier_frame_1_pts_2d.shape)
    #print("inlier_mask", inlier_mask.shape)
    #print("frame_1_kp", len(frame_1_kp))
    #print("frame_1_kp_des", frame_1_kp_des.shape)
    
    for m in inlier_matches:#[camera_1_inlier_mask]:
        old_idx = m[0].queryIdx
        new_idx = m[0].trainIdx
        
        inliers_frame_1_kp.append(frame_1_kp[old_idx])
        inliers_frame_2_kp.append(frame_2_kp[new_idx])
        
        inliers_frame_1_kp_des.append(frame_1_kp_des[old_idx])
        inliers_frame_2_kp_des.append(frame_2_kp_des[new_idx])
        
    inliers_frame_1_kp = tuple(inliers_frame_1_kp)
    inliers_frame_2_kp = tuple(inliers_frame_2_kp)
    
    #print("inliers_frame_1_kp", len(inliers_frame_1_kp))
    #print("inliers_frame_2_kp", len(inliers_frame_2_kp))
    
    inliers_frame_1_kp_des = np.asarray(inliers_frame_1_kp_des)
    inliers_frame_2_kp_des = np.asarray(inliers_frame_2_kp_des)
    
    #print("inliers_frame_1_kp_des", inliers_frame_1_kp_des.shape)
    #print("inliers_frame_2_kp_des", inliers_frame_2_kp_des.shape)
    
    
    #print("2")
    #print("frame_1_kp", frame_1_kp, type(frame_1_kp), "inlier_mask", inlier_mask, "frame_1_pts_2d", frame_1_pts_2d.shape, type(frame_1_pts_2d), "good_matches", len(good_matches))
    #print("lens", len(frame_1_kp), len(inlier_mask))
    
    #inliers_frame_1_kp = frame_1_kp[inlier_mask]
    #inliers_frame_2_kp = frame_2_kp[inlier_mask]
    #inliers_frame_1_kp_des = frame_1_kp_des[inlier_mask]
    #inliers_frame_2_kp_des = frame_2_kp_des[inlier_mask]
    return F, mask_fundamental, inlier_frame_1_pts_2d, inlier_frame_2_pts_2d, inliers_frame_1_kp, inliers_frame_2_kp, inliers_frame_1_kp_des, inliers_frame_2_kp_des, inlier_matches, outlier_matches, inlier_mask

def obtain_fundamental_and_inliers_with_ransac_2d(
    frame_1_pts_2d: np.ndarray[np.float32],
    frame_2_pts_2d: np.ndarray[np.float32],
    ransac_reproj_threshold: float = 3.0,
    confidence: float = 0.99,
    max_iters: int = 2000
):
    # Estimate fundamental matrix using RANSAC
    fundamental_result: tuple[cv2.typing.MatLike, cv2.typing.MatLike] = cv2.findFundamentalMat(frame_1_pts_2d, frame_2_pts_2d, method=cv2.FM_RANSAC, ransacReprojThreshold=ransac_reproj_threshold, confidence=confidence, maxIters=max_iters)
    F, mask_fundamental = fundamental_result
    if mask_fundamental is None:
        return None, None, None
    inlier_mask: Type[np.ndarray[bool]] = mask_fundamental.ravel() == 1
    return F, mask_fundamental, inlier_mask
    
def main():
    steps: List[StepType] = []
    calibration = load_camera_calibration()
    
    camera_indicies = []
    for camera_index, _ in calibration.items():
        camera_indicies.append(camera_index)
        
    cam1 = cv2.VideoCapture(camera_indicies[0])
    cam2 = cv2.VideoCapture(camera_indicies[1])

    n_features = 1000
    factor = 1.2
    orb = cv2.ORB_create(n_features, factor)
    bf_matcher = get_matcher()
    prev_time = time.time()
    
    while True:
        frame1_img = rectify_image(get_frame(cam1), calibration[camera_indicies[0]])
        frame2_img = rectify_image(get_frame(cam2), calibration[camera_indicies[1]])

        if frame1_img is None or frame2_img is None:
            continue

        frame_1_kp, frame_1_kp_des = get_orb_keypoint_and_descriptors(orb, frame1_img)
        frame_2_kp, frame_2_kp_des = get_orb_keypoint_and_descriptors(orb, frame2_img)
        
        # Extract pixel coordinates of matched keypoints
        stereo_2d_pts_result = get_stereo_2d_correspondance(bf_matcher, frame_1_kp, frame_2_kp, frame_1_kp_des, frame_2_kp_des)
        good_matches, frame_1_pts_2d, frame_2_pts_2d = stereo_2d_pts_result
        
        if good_matches is None:
            continue
        
        ransac_filtered_correspondances_result = filter_stereo_correspondences_with_ransac(frame_1_kp, frame_2_kp, frame_1_kp_des, frame_1_kp_des, frame_1_pts_2d, frame_2_pts_2d, good_matches)
        F, mask_fundamental, inlier_frame_1_pts_2d, inlier_frame_2_pts_2d, inliers_frame_1_kp, inliers_frame_2_kp, inliers_frame_1_kp_des, inliers_frame_2_kp_des, inlier_matches, outlier_matches, inlier_mask = ransac_filtered_correspondances_result
        if F is None:
            print("ignore F")
            continue
        final_img: Type[cv2.typing.MatLike] = cv2.drawMatchesKnn(frame1_img,frame_1_kp,frame2_img,frame_2_kp,good_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Display FPS on the frame   
        current_time = time.time()
        elapsed_time = current_time - prev_time
        fps = 1.0 / elapsed_time
        prev_time = current_time
        cv2.putText(final_img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("stereo", final_img)
        k = cv2.waitKey(1)
        print("SHOWING STEREO!!!")
        
        if mask_fundamental is None:
            print("ignore fundamental")
            continue
        
        # Draw lines connecting inliers / outliers on both images
        for idx in range(len(inlier_frame_1_pts_2d)):
            p1 = (int(inlier_frame_1_pts_2d[idx][0]), int(inlier_frame_1_pts_2d[idx][1]))
            p2 = (int(inlier_frame_2_pts_2d[idx][0] + frame1_img.shape[1]), int(inlier_frame_2_pts_2d[idx][1]))
            final_img = cv2.line(final_img, p1, p2, (0, 255, 0), 2)
        for m in outlier_matches:
            pt1 = tuple(map(int, frame_1_kp[m[0].queryIdx].pt))
            pt2 = tuple(map(int, frame_2_kp[m[0].trainIdx].pt))
            final_img = cv2.line(final_img, pt1, (pt2[0] + frame1_img.shape[1], pt2[1]), (0, 0, 255), 2)
        cv2.imshow("stereo", final_img)
        k = cv2.waitKey(1)
        
        if len(steps) == 0:
            # initial
            # define camera 1 as being at zero,zero pose
            R1 = np.eye(3)
            T1 = np.zeros((3, 1))
            P1 = np.hstack((R1, T1))
            
            # recover focal length
            
            # recover pose for 2nd camera
            try:
                n_pose_inliers, R2, T2, pose_mask = cv2.recoverPose(F, frame_1_pts_2d[inlier_mask], frame_2_pts_2d[inlier_mask])  #cv2.recoverPose(F, inlier_frame_1_pts_2d, inlier_frame_2_pts_2d)
            except Exception as e:
                print("error with recover pose", e)
                continue
            
            if n_pose_inliers < 30:
                print("ignore n_pose_inliers", n_pose_inliers)
                continue
            
            print("R1", R1)
            print("T1", T1)
            print("R2", R2)
            print("T2", T2)
            
            P2 = np.hstack((R2, T2))
            
            # might want to mask again! pose_mask checkme
            
            # ok so now we have both cameras poses we need to get the 3d points triangulated into world space
            
            pts_3d_in_world_space = triangulate_points(calibration[camera_indicies[0]]["camera_matrix"], R1, T1, calibration[camera_indicies[1]]["camera_matrix"], R2, T2, inlier_frame_1_pts_2d.reshape(-1, 2).T, inlier_frame_2_pts_2d.reshape(-1, 2).T)
            
            #pts_3d_in_world_space = triangulate_points(P1,P2,inlier_frame_1_pts_2d.T, inlier_frame_2_pts_2d.T)
            
            step: StepType = {
                "frame1_img": frame1_img,
                "frame2_img": frame2_img,
                "frame_1_kp": inliers_frame_1_kp,
                "frame_2_kp": inliers_frame_2_kp,
                "frame_1_kp_des": inliers_frame_1_kp_des,
                "frame_2_kp_des": inliers_frame_2_kp_des,
                "frame_1_pts_2d": inlier_frame_1_pts_2d,
                "frame_2_pts_2d": inlier_frame_2_pts_2d,
                "frame_1_R": R1,
                "frame_1_T": T1,
                "frame_2_R": R2,
                "frame_2_T": T2,
                "frame_1_P": P1,
                "frame_2_P": P2,
                "pts_3d": pts_3d_in_world_space
            }
            
            steps.append(step)
            visualize_all3(calibration, camera_indicies, steps)
        else:
            # deal with other situations... ignore for now
            prev_step = steps[len(steps)-1]
            
            # the goal of this step is to do pnp matching in order to do this we must work out which features in our new frames,
            # correspond to each inlier feature of the last frame
            
            # get stereo matches for each frame pair between subsequent time steps
            camera_1_2d_pts_result = get_stereo_2d_correspondance(bf_matcher, prev_step["frame_1_kp"], frame_1_kp, prev_step["frame_1_kp_des"], frame_1_kp_des)
            camera_1_good_matches, frame_1_old_pts_2d, frame_1_new_pts_2d = camera_1_2d_pts_result
            
            camera_2_2d_pts_result = get_stereo_2d_correspondance(bf_matcher, prev_step["frame_2_kp"], frame_2_kp, prev_step["frame_2_kp_des"], frame_2_kp_des)
            camera_2_good_matches, frame_2_old_pts_2d, frame_2_new_pts_2d = camera_2_2d_pts_result
            
            if camera_1_good_matches is None or camera_2_good_matches is None:
                continue
            
            # geometry check
            
            #camera_1_ret = obtain_fundamental_and_inliers_with_ransac_2d(frame_1_old_pts_2d, frame_1_new_pts_2d)
            #camera_1_F, camera_1_mask_fundamental, camera_1_inlier_mask = camera_1_ret
            
            #camera_2_ret = obtain_fundamental_and_inliers_with_ransac_2d(frame_2_old_pts_2d, frame_2_new_pts_2d)
            #camera_2_F, camera_2_mask_fundamental, camera_2_inlier_mask = camera_2_ret
            
            #if camera_1_mask_fundamental is None or camera_2_mask_fundamental is None:
            #    continue
            
            # now take the good matches for each of the cameras and recover the 3d map points
            camera_1_old_pts_3d = []
            camera_1_new_pts_2d = frame_1_old_pts_2d#[]
            for m in np.array(camera_1_good_matches):#[camera_1_inlier_mask]:
                old_idx = m[0].queryIdx
                #new_idx = m[0].trainIdx
                frame_1_old_map_point = prev_step["pts_3d"][old_idx]
                #frame_1_new_pt_2d = frame_1_new_pts_2d[new_idx]
                camera_1_old_pts_3d.append(frame_1_old_map_point)
                #camera_1_new_pts_2d.append(frame_1_new_pt_2d)
            camera_1_old_pts_3d = np.array(camera_1_old_pts_3d)
            #camera_1_new_pts_2d = np.array(camera_1_new_pts_2d)

            camera_2_old_pts_3d = []
            camera_2_new_pts_2d = frame_2_old_pts_2d#[]
            for m in np.array(camera_2_good_matches):#[camera_2_inlier_mask]:
                old_idx = m[0].queryIdx
                #new_idx = m[0].trainIdx
                frame_2_old_map_point = prev_step["pts_3d"][old_idx]
                #frame_2_new_pt_2d = frame_2_new_pts_2d[new_idx]
                camera_2_old_pts_3d.append(frame_2_old_map_point)
                #camera_2_new_pts_2d.append(frame_2_new_pt_2d)
            camera_2_old_pts_3d = np.array(camera_2_old_pts_3d)
            #camera_2_new_pts_2d = np.array(camera_2_new_pts_2d)
                        
            # Use the new camera matrices obtained during rectification
            h1, w1 = frame1_img.shape[:2]
            h2, w2 = frame2_img.shape[:2]
            undistorted_camera_1_matrix, _ = cv2.getOptimalNewCameraMatrix(calibration[camera_indicies[0]]["camera_matrix"], calibration[camera_indicies[0]]["distortion_coeffs"], (w1, h1), 1, (w1, h1))
            undistorted_camera_2_matrix2, _ = cv2.getOptimalNewCameraMatrix(calibration[camera_indicies[1]]["camera_matrix"], calibration[camera_indicies[1]]["distortion_coeffs"], (w2, h2), 1, (w2, h2))

            # PnP for both frames using the undistorted camera matrices
            print("camera_1_good_matches", len(camera_1_good_matches))
            print("camera_2_good_matches", len(camera_2_good_matches))
            print("camera_1_old_pts_3d", len(camera_1_old_pts_3d))
            print("camera_1_new_pts_2d", len(camera_1_new_pts_2d))
            print("camera_2_old_pts_3d", len(camera_2_old_pts_3d))
            print("camera_2_new_pts_2d", len(camera_2_new_pts_2d))
            
            if (len(camera_1_old_pts_3d) < 10 or len(camera_2_old_pts_3d) < 10):
                continue

            camera_1_pnp_ransac_success, R1_vec, T1_vec, camera_1_pnp_ransac_inliers_mask = cv2.solvePnPRansac(camera_1_old_pts_3d, camera_1_new_pts_2d, undistorted_camera_1_matrix, None)
            camera_2_pnp_ransac_success, R2_vec, T2_vec, camera_2_pnp_ransac_inliers_mask = cv2.solvePnPRansac(camera_2_old_pts_3d, camera_2_new_pts_2d, undistorted_camera_2_matrix2, None)

            if not camera_1_pnp_ransac_success or not camera_2_pnp_ransac_success:
                continue
            
            print("camera_1_pnp_ransac_success", camera_1_pnp_ransac_success)
            print("R1_vec", R1_vec)
            print("T1_vec", T1_vec)
            print("R2_vec", R2_vec)
            print("T2_vec", T2_vec)
            
            #R1, _ = cv2.Rodrigues(R1)
            #R2, _ = cv2.Rodrigues(R2)
            #print("R1", R1)
            #print("R2", R2)

            # Refine camera 1 pose using solvePnPRefineLM
            pnp_refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            camera_1_refine_result = cv2.solvePnPRefineLM(
                camera_1_old_pts_3d[camera_1_pnp_ransac_inliers_mask],
                camera_1_new_pts_2d[camera_1_pnp_ransac_inliers_mask],
                undistorted_camera_1_matrix,
                None,
                R1_vec,
                T1_vec,
                criteria=pnp_refine_criteria
            )
            print ("camera_1_refine_result", camera_1_refine_result)
            R1_vec, T1_vec = camera_1_refine_result

            # Refine camera 2 pose using solvePnPRefineLM
            R2_vec, T2_vec = cv2.solvePnPRefineLM(
                camera_2_old_pts_3d[camera_2_pnp_ransac_inliers_mask],
                camera_2_new_pts_2d[camera_2_pnp_ransac_inliers_mask],
                undistorted_camera_2_matrix2,
                None,
                R2_vec,
                T2_vec,
                criteria=pnp_refine_criteria
            )
            
            R1, _ = cv2.Rodrigues(R1_vec)
            R2, _ = cv2.Rodrigues(R2_vec)

            # Create overall pose
            P1 = np.hstack((R1, T1))
            P2 = np.hstack((R2, T2))
            
            print("about to triangulate")
            print("P1", P1)
            print("P2", P2)
            
            #pts_3d_in_world_space = triangulate_points(P1,P2,inlier_frame_1_pts_2d.reshape(-1, 2).T, inlier_frame_2_pts_2d.reshape(-1, 2).T)
            pts_3d_in_camera_1_space = triangulate_points(calibration[camera_indicies[0]]["camera_matrix"], R1, T1, calibration[camera_indicies[1]]["camera_matrix"], R2, T2, inlier_frame_1_pts_2d.reshape(-1, 2).T, inlier_frame_2_pts_2d.reshape(-1, 2).T)
            pts_3d_in_world_space = transform_points_to_origin(pts_3d_in_camera_1_space, R1, T1)
            # fixme check that these points are in world space and do not need to be reoriented by the pose of camera 1
            
            step: StepType = {
                "frame1_img": frame1_img,
                "frame2_img": frame2_img,
                "frame_1_kp": inliers_frame_1_kp,
                "frame_2_kp": inliers_frame_2_kp,
                "frame_1_kp_des": inliers_frame_1_kp_des,
                "frame_2_kp_des": inliers_frame_2_kp_des,
                "frame_1_pts_2d": inlier_frame_1_pts_2d,
                "frame_2_pts_2d": inlier_frame_2_pts_2d,
                "frame_1_R": R1,
                "frame_1_T": T1,
                "frame_2_R": R2,
                "frame_2_T": T2,
                "frame_1_P": P1,
                "frame_2_P": P2,
                "pts_3d": pts_3d_in_world_space
            }
            
            steps.append(step)
            
            visualize_all3(calibration, camera_indicies, steps)
        
        cv2.imshow("stereo", final_img)    
        if k%256 == 27: # Escape
            break
    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
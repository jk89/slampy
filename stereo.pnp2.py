from calib import detect_camera_indicies, load_camera_calibration, auto_load_or_calibrate_cameras
import cv2
import numpy as np
import time 
import open3d as o3d
from typing import Type, List, Sequence, TypedDict, Dict

WIDTH = 1280
HEIGHT = 720

class Index_2d_des_3d(TypedDict):
    frame1: Dict[int, int]
    frame2: Dict[int, int]

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
    stereo_matches: Sequence[cv2.DMatch]
    index_2d_des_3d: Index_2d_des_3d


def visualize_all3(visualizer, calibration, camera_indicies, steps: List[StepType], wait_for_frame = True):
    # Step 0 - Init
    rainbow_colors = [(1.0, 0.0, 0.0), (1.0, 0.5, 0.0), (1.0, 1.0, 0.0), (0.5, 1.0, 0.0), (0.0, 1.0, 0.0),
                 (0.0, 1.0, 0.5), (0.0, 1.0, 1.0), (0.0, 0.5, 1.0), (0.0, 0.0, 1.0), (0.5, 0.0, 1.0),
                 (1.0, 0.0, 1.0)]

    # Step 1 - Get scene origin sphere
    sphere_origin = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
    sphere_origin.translate([0.0, 0.0, 0.0])

    # Step 2 - Create visualizer object
    if wait_for_frame is True:
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window(width=WIDTH, height=HEIGHT)

    # Step 3 - Add objects to visualizer
    #visualizer.add_geometry(sphere_origin)

    # Combine all 3D points from each step
    #all_points = np.concatenate([step["pts_3d"] for step in steps])

    # Visualize all 3D points
    

    for idx, step in enumerate(steps):
        
        color_index = idx % len(rainbow_colors)

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(step["pts_3d"])
        #print("[rainbow_colors[color_index]] * len(steps)", [rainbow_colors[color_index]] * len(steps))
        point_cloud.colors = o3d.utility.Vector3dVector([rainbow_colors[color_index]] * len(step["pts_3d"]))

        visualizer.add_geometry(point_cloud, reset_bounding_box=True)
        
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

        sphere_cam_1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        sphere_cam_1.paint_uniform_color(rainbow_colors[color_index])  # Set the color to red
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

        sphere_cam_2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        sphere_cam_2.paint_uniform_color(rainbow_colors[color_index])  # Set the color to blue
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
    if frame is None:
        return None
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

def get_stereo_2d_correspondance(bf_matcher:Type[cv2.BFMatcher], frame_1_kp_des, frame_2_kp_des):
    try:
        matches = get_covisible_features(bf_matcher, frame_1_kp_des, frame_2_kp_des)
    except Exception as e:
        print("e", e, frame_1_kp_des.shape, frame_2_kp_des.shape)
        return []
        
    lowe_matches:Sequence[cv2.DMatch] = []
    
    #print("len matches", len(matches))
                
    # Lowe's test
    for x in matches:
        if len(x) > 1:
            m,n = x
            if m.distance < 0.8*n.distance: #75 #here
                lowe_matches.append(m)
                
    #print("lowe_matches", len(lowe_matches))
        
    return lowe_matches

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


def transform_points_to_origin_2(pts_3d, R1, T1):
    pts_3d_at_origin = np.dot(pts_3d, R1.T)
    pts_3d_at_origin += T1.T 
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

def filter_stereo_kp_des_with_lowe_matches(
    lowe_matches: Sequence[cv2.DMatch],
    frame_1_kp: Sequence[cv2.KeyPoint],
    frame_2_kp: Sequence[cv2.KeyPoint],
    frame_1_kp_des: cv2.typing.MatLike,
    frame_2_kp_des: cv2.typing.MatLike):
    
    lowe_frame_1_kp = []
    lowe_frame_2_kp = []
    lowe_frame_1_kp_des = []
    lowe_frame_2_kp_des = []
    
    for m in lowe_matches:
        old_idx = m.queryIdx
        new_idx = m.trainIdx
        lowe_frame_1_kp.append(frame_1_kp[old_idx])
        lowe_frame_2_kp.append(frame_2_kp[new_idx])
        lowe_frame_1_kp_des.append(frame_1_kp_des[old_idx])
        lowe_frame_2_kp_des.append(frame_2_kp_des[new_idx])
        
    lowe_frame_1_kp = tuple(lowe_frame_1_kp)
    lowe_frame_2_kp = tuple(lowe_frame_2_kp)
    lowe_frame_1_kp_des = np.asarray(lowe_frame_1_kp_des)
    lowe_frame_2_kp_des = np.asarray(lowe_frame_2_kp_des)
    
    return lowe_frame_1_kp, lowe_frame_2_kp, lowe_frame_1_kp_des, lowe_frame_2_kp_des
    
def filter_stereo_correspondences_with_inliers_mask(
    frame_1_kp: Sequence[cv2.KeyPoint],
    frame_2_kp: Sequence[cv2.KeyPoint],
    frame_1_kp_des: cv2.typing.MatLike,
    frame_2_kp_des: cv2.typing.MatLike,
    frame_1_pts_2d: np.ndarray[np.float32],
    frame_2_pts_2d: np.ndarray[np.float32],
    inlier_mask: Type[np.ndarray[bool] | None] = None
    ):
    
    outlier_mask: Type[np.ndarray[bool]] = ~inlier_mask
    inlier_matches: np.ndarray[cv2.DMatch] = np.array(lowe_matches)[inlier_mask]
    outlier_matches: np.ndarray[cv2.DMatch] = np.array(lowe_matches)[outlier_mask]
    
    inliers_frame_1_kp = []
    inliers_frame_2_kp = []
    inliers_frame_1_kp_des = []
    inliers_frame_2_kp_des = []
    inlier_frame_1_pts_2d: np.ndarray[np.float32] = frame_1_pts_2d[inlier_mask]
    inlier_frame_2_pts_2d: np.ndarray[np.float32] = frame_2_pts_2d[inlier_mask]
    
    for m in inlier_matches:
        old_idx = m[0].queryIdx
        new_idx = m[0].trainIdx
        
        inliers_frame_1_kp.append(frame_1_kp[old_idx])
        inliers_frame_2_kp.append(frame_2_kp[new_idx])
        
        inliers_frame_1_kp_des.append(frame_1_kp_des[old_idx])
        inliers_frame_2_kp_des.append(frame_2_kp_des[new_idx])
        
    inliers_frame_1_kp = tuple(inliers_frame_1_kp)
    inliers_frame_2_kp = tuple(inliers_frame_2_kp)
    
    inliers_frame_1_kp_des = np.asarray(inliers_frame_1_kp_des)
    inliers_frame_2_kp_des = np.asarray(inliers_frame_2_kp_des)

    return inlier_matches, outlier_matches, inlier_frame_1_pts_2d, inlier_frame_2_pts_2d, inliers_frame_1_kp, inliers_frame_2_kp, inliers_frame_1_kp_des, inliers_frame_2_kp_des


def normalise_pts_2d(points, K, D):
    # Undistort the points using the camera matrix (K) and distortion coefficients (D)
    undistorted_points = cv2.undistortPoints(points, K, D)

    # Normalize by dividing by the focal length
    normalized_points = undistorted_points / K[0, 0]

    return normalized_points

def normalize_keypoints(kp_list, K, D):
    # Convert the keypoints to a numpy array
    points_array = np.array([kp.pt for kp in kp_list], dtype=np.float32)

    # Undistort the points using the camera matrix (K) and distortion coefficients (D)
    undistorted_points = cv2.undistortPoints(points_array, K, D)

    # Normalize by dividing by the focal length
    normalized_points = undistorted_points / K[0, 0]
    
    # Create a list of cv2.KeyPoint objects with normalized coordinates
    normalized_keypoints_list = [cv2.KeyPoint(x=point[0][0], y=point[0][1], size=kp.size, angle=kp.angle,
                                               response=kp.response, octave=kp.octave, class_id=kp.class_id)
                                  for point, kp in zip(normalized_points, kp_list)]

    return normalized_keypoints_list

# from_2d_correspondances_with_ransac

def obtain_fundamental_and_inliers_from_2d_correspondances_with_ransac(
    frame_1_pts_2d: np.ndarray[np.float32],
    frame_2_pts_2d: np.ndarray[np.float32],
    ransac_reproj_threshold: float = 2.0,
    confidence: float = 0.99,
    max_iters: int = 2000
):
    # Estimate fundamental matrix using RANSAC
    fundamental_result: tuple[cv2.typing.MatLike, cv2.typing.MatLike] = cv2.findFundamentalMat(frame_1_pts_2d, frame_2_pts_2d, method=cv2.FM_RANSAC, ransacReprojThreshold=ransac_reproj_threshold, confidence=confidence, maxIters=max_iters)
    F, mask_fundamental = fundamental_result
    
    if mask_fundamental is None:
        return None
    
    inlier_mask: Type[np.ndarray[bool]] = mask_fundamental.ravel() == 1
    
    #print("inlier_mask", mask_fundamental.ravel())
    outlier_mask: Type[np.ndarray[bool]] = ~inlier_mask
    
    return F, mask_fundamental, inlier_mask, outlier_mask


def filter_keypoints_and_descriptors(matches, kp1, kp2, kp1_des, kp2_des):
    # Get unique indices from matches
    query_indices = sorted(set(match.queryIdx for match in matches))
    train_indices = sorted(set(match.trainIdx for match in matches))

    # Filter keypoints
    filtered_kp1 = [kp1[i] for i in query_indices]
    filtered_kp2 = [kp2[i] for i in train_indices]

    # Filter descriptors
    filtered_kp1_des = kp1_des[query_indices]#[kp1_des[i] for i in query_indices]
    filtered_kp2_des = kp2_des[train_indices]#[kp2_des[i] for i in train_indices]

    # Create a mapping from old indices to new indices for kp1 and kp2
    index_mapping_kp1 = {old_index: new_index for new_index, old_index in enumerate(query_indices)}
    index_mapping_kp2 = {old_index: new_index for new_index, old_index in enumerate(train_indices)}

    # Create new matches using the updated keypoints
    converted_matches = [cv2.DMatch(index_mapping_kp1[match.queryIdx], index_mapping_kp2[match.trainIdx], match.distance)
                   for match in matches]

    return filtered_kp1, filtered_kp2, filtered_kp1_des, filtered_kp2_des, converted_matches

#filtered_kp1, filtered_kp2, filtered_kp1_des, filtered_kp2_des, new_matches = filter_keypoints_and_descriptors(matches, kp1, kp2, kp1_des, kp2_des)##
    
def main():
    steps: List[StepType] = []
    calibration = auto_load_or_calibrate_cameras()
    
    print("calibration", calibration)
    
    camera_indicies = []
    for camera_index, _ in calibration.items():
        camera_indicies.append(camera_index)
    
    if len(camera_indicies) < 2:
        print("To few cameras",len(camera_indicies))
        return
        
    cam1 = cv2.VideoCapture(camera_indicies[0])
    cam2 = cv2.VideoCapture(camera_indicies[1])

    n_features = 1000
    factor = 1.2
    orb = cv2.ORB_create(n_features, factor)
    bf_matcher = get_matcher()
    prev_time = time.time()
    
    K1 = calibration[camera_indicies[0]]["camera_matrix"]
    K2 = calibration[camera_indicies[1]]["camera_matrix"]
    D1 = calibration[camera_indicies[0]]["distortion_coeffs"]
    D2 = calibration[camera_indicies[1]]["distortion_coeffs"]
    
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=WIDTH, height=HEIGHT)
    wait_for_frame = True
    
    while True:
        frame1_img = rectify_image(get_frame(cam1), calibration[camera_indicies[0]])
        frame2_img = rectify_image(get_frame(cam2), calibration[camera_indicies[1]])

        if frame1_img is None or frame2_img is None:
            print("ignore frames are None")
            continue

        frame_1_kp, frame_1_kp_des = get_orb_keypoint_and_descriptors(orb, frame1_img)
        if len(frame_1_kp) == 0:
            print("ignore len(frame_1_kp) == 0")
            continue
        frame_2_kp, frame_2_kp_des = get_orb_keypoint_and_descriptors(orb, frame2_img)
        if len(frame_2_kp) == 0:
            print("ignore len(frame_2_kp) == 0")
            continue
        
        #normalize_keypoints
        frame_1_kp_norm = normalize_keypoints(frame_1_kp, K1, D1 )
        frame_2_kp_norm = normalize_keypoints(frame_2_kp, K2, D2 )
        
        # Extract pixel coordinates of matched keypoints
        lowe_matches = get_stereo_2d_correspondance(bf_matcher, frame_1_kp_des, frame_2_kp_des)
        
        if len(lowe_matches) == 0:
            print("ignore lowe")
            continue
        
        #frame_1_kp_norm, frame_2_kp_norm, frame_1_kp_des, frame_2_kp_des  = filter_stereo_kp_des_with_lowe_matches(lowe_matches, frame_1_kp_norm, frame_2_kp_norm, frame_1_kp_des, frame_2_kp_des)
        
        # Display keypoing matches on the frame  
        final_img: Type[cv2.typing.MatLike] = cv2.drawMatchesKnn(frame1_img,frame_1_kp,frame2_img,frame_2_kp,[[i] for i in lowe_matches],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Display FPS on the frame   
        current_time = time.time()
        elapsed_time = current_time - prev_time
        fps = 1.0 / elapsed_time
        prev_time = current_time
        cv2.putText(final_img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("stereo", final_img)
        k = cv2.waitKey(1)
        if k == 32: 
            wait_for_frame = not wait_for_frame
        #print("SHOWING STEREO!!!")          
        
        frame_1_pts_2d :np.ndarray[(Ellipsis, 2), np.float32] = np.float32([frame_1_kp[m.queryIdx].pt for m in lowe_matches]).reshape(-1, 2)
        frame_2_pts_2d: np.ndarray[(Ellipsis, 2), np.float32] = np.float32([frame_2_kp[m.trainIdx].pt for m in lowe_matches]).reshape(-1, 2)
        frame_1_pts_2d_norm :np.ndarray[(Ellipsis, 2), np.float32] = np.float32([frame_1_kp_norm[m.queryIdx].pt for m in lowe_matches]).reshape(-1, 2)
        frame_2_pts_2d_norm: np.ndarray[(Ellipsis, 2), np.float32] = np.float32([frame_2_kp_norm[m.trainIdx].pt for m in lowe_matches]).reshape(-1, 2)

        
        # only lowe good tested good 2d keypoints points should be reaching this function
        # get fundamental, inlier and outlier masks
        
        fundamental_result = obtain_fundamental_and_inliers_from_2d_correspondances_with_ransac(frame_1_pts_2d, frame_2_pts_2d) #frame_1_pts_2d_norm
        if fundamental_result is None:
            print("ignore F")
            continue
        F, mask_fundamental, inlier_mask, outlier_mask = fundamental_result

        frame_1_pts_2d_epipolar = frame_1_pts_2d[inlier_mask]
        frame_2_pts_2d_epipolar = frame_2_pts_2d[inlier_mask]
        
        frame_1_pts_2d_epipolar_outliers = frame_1_pts_2d[outlier_mask]
        frame_2_pts_2d_epipolar_outliers = frame_2_pts_2d[outlier_mask]
        
        
        
        #print("outlier_mask", outlier_mask)
        print("len frame_1_pts_2d_epipolar", len(frame_1_pts_2d_epipolar))
        print("len frame_2_pts_2d_epipolar", len(frame_2_pts_2d_epipolar))
        #print("len frame_1_pts_2d_epipolar_outliers", len(frame_1_pts_2d_epipolar_outliers))
        #print("len frame_2_pts_2d_epipolar_outliers", len(frame_2_pts_2d_epipolar_outliers))

        #ransac_filtered_correspondances_result = filter_stereo_correspondences_with_inliers_mask(frame_1_kp, frame_2_kp, frame_1_kp_des, frame_1_kp_des, frame_1_pts_2d, frame_2_pts_2d, lowe_matches)
        #F, mask_fundamental, inlier_frame_1_pts_2d, inlier_frame_2_pts_2d, inliers_frame_1_kp, inliers_frame_2_kp, inliers_frame_1_kp_des, inliers_frame_2_kp_des, inlier_matches, outlier_matches, inlier_mask = ransac_filtered_correspondances_result


        # Draw lines connecting inliers / outliers on both images
        for idx in range(len(frame_1_pts_2d_epipolar)):
            p1 = (int(frame_1_pts_2d_epipolar[idx][0]), int(frame_1_pts_2d_epipolar[idx][1]))
            p2 = (int(frame_2_pts_2d_epipolar[idx][0] + frame1_img.shape[1]), int(frame_2_pts_2d_epipolar[idx][1]))
            final_img = cv2.line(final_img, p1, p2, (0, 255, 0), 2)

        for idx in range(len(frame_1_pts_2d_epipolar_outliers)):
            frame_1_outlier_pts_2d = frame_1_pts_2d_epipolar_outliers[idx]
            frame_2_outlier_pts_2d = frame_2_pts_2d_epipolar_outliers[idx]
            pt1 = tuple(map(int, frame_1_outlier_pts_2d))
            pt2 = tuple(map(int, frame_2_outlier_pts_2d))
            final_img = cv2.line(final_img, pt1, (pt2[0] + frame1_img.shape[1], pt2[1]), (0, 0, 255), 2)
        cv2.imshow("stereo", final_img)
        k = cv2.waitKey(1)
        if k == 32:
            wait_for_frame = not wait_for_frame
        
        if len(steps) == 0:
            # initial
            # define camera 1 as being at zero,zero pose
            R1 = np.eye(3)
            T1 = np.zeros((3, 1))
            P1 = np.hstack((R1, T1))
            
            # recover pose for 2nd camera
            try:
                E = np.dot(np.dot(np.transpose(K2), F), K1)
                recover_pose_result = cv2.recoverPose(frame_1_pts_2d_epipolar, frame_2_pts_2d_epipolar, K1, D1, K2, D2, E, prob = 0.99999, threshold = 0.6)  #cv2.recoverPose(F, inlier_frame_1_pts_2d, inlier_frame_2_pts_2d)
                n_pose_inliers, E, R2, T2, pose_mask = recover_pose_result
                pose_mask = np.array(pose_mask).flatten() == 1
            except Exception as e:
                print("ignore error with recover pose", e)
                continue
            
            print("n_pose_inliers", n_pose_inliers)
            if n_pose_inliers < 150: # here
                print("ignore n_pose_inliers")
                continue
            
            #print("R1", R1)
            #print("T1", T1)
            #print("R2", R2)
            #print("T2", T2)
            
            P2 = np.hstack((R2, T2))
            # [pose_mask]
            #print("pose_mask", pose_mask)
            frame_1_kp_filtered, frame_2_kp_filtered, frame_1_kp_des_filtered, frame_2_kp_des_filtered, converted_matches = filter_keypoints_and_descriptors(list(np.array(lowe_matches)[inlier_mask][pose_mask].reshape(-1)), frame_1_kp, frame_2_kp, frame_1_kp_des, frame_2_kp_des)
            
            inlier_frame_1_pts_2d = np.float32([frame_1_kp_filtered[m.queryIdx].pt for m in converted_matches]).reshape(-1, 2)
            inlier_frame_2_pts_2d = np.float32([frame_2_kp_filtered[m.trainIdx].pt for m in converted_matches]).reshape(-1, 2)
            
            pts_3d_in_world_space = triangulate_points(calibration[camera_indicies[0]]["camera_matrix"], R1, T1, calibration[camera_indicies[1]]["camera_matrix"], R2, T2, inlier_frame_1_pts_2d.reshape(-1, 2).T, inlier_frame_2_pts_2d.reshape(-1, 2).T)
            
            # need to make an index for these world points
            # as in the next frame we will be trying to get them from a keypoint index
            # we get des idx from a particular camera and want to be able to find what map point it relates to
            #print("pts_3d_in_world_space", pts_3d_in_world_space.shape)
            #print("np.array(lowe_matches)[inlier_mask][pose_mask]", len(np.array(lowe_matches)[inlier_mask][pose_mask]), n_pose_inliers)
            frame_1_kp_des_pt_3d_index = {}
            frame_2_kp_des_pt_3d_index = {}
            for idx_3d_pt, m in enumerate(converted_matches):
                exists = pts_3d_in_world_space[idx_3d_pt]
                old_idx = m.queryIdx
                frame_1_kp_des_pt_3d_index[old_idx] = idx_3d_pt
                new_idx = m.trainIdx
                frame_2_kp_des_pt_3d_index[new_idx] = idx_3d_pt
                
            step: StepType = {
                "frame1_img": frame1_img,
                "frame2_img": frame2_img,
                "frame_1_kp": frame_1_kp_filtered,
                "frame_2_kp": frame_2_kp_filtered,
                "frame_1_kp_des": frame_1_kp_des_filtered,
                "frame_2_kp_des": frame_2_kp_des_filtered,
                "frame_1_pts_2d": inlier_frame_1_pts_2d,
                "frame_2_pts_2d": inlier_frame_2_pts_2d,
                "frame_1_R": R1,
                "frame_1_T": T1,
                "frame_2_R": R2,
                "frame_2_T": T2,
                "frame_1_P": P1,
                "frame_2_P": P2,
                "pts_3d": pts_3d_in_world_space,
                "stereo_matches": converted_matches,
                "index_2d_des_3d": {
                    "frame1":frame_1_kp_des_pt_3d_index,
                    "frame2":frame_2_kp_des_pt_3d_index
                }
            }
            
            steps.append(step)
            visualize_all3(visualizer, calibration, camera_indicies, steps, wait_for_frame)
        else:
            # deal with other situations... ignore for now
            prev_step = steps[len(steps)-1]
            
            # filter out the values which do not yield good 3d points
            frame_1_kp_filtered, frame_2_kp_filtered, frame_1_kp_des_filtered, frame_2_kp_des_filtered, converted_matches = filter_keypoints_and_descriptors(list(np.array(lowe_matches)[inlier_mask].reshape(-1)), frame_1_kp, frame_2_kp, frame_1_kp_des, frame_2_kp_des)
            
            inlier_frame_1_pts_2d = np.float32([frame_1_kp_filtered[m.queryIdx].pt for m in converted_matches]).reshape(-1, 2)
            inlier_frame_2_pts_2d = np.float32([frame_2_kp_filtered[m.trainIdx].pt for m in converted_matches]).reshape(-1, 2)
            
            # the goal of this step is to do pnp matching in order to do this we must work out which features in our new frames,
            # correspond to each inlier feature of the last frame
            
            # get stereo matches for each frame pair between subsequent time steps
            camera_1_lowe_matches = get_stereo_2d_correspondance(bf_matcher, prev_step["frame_1_kp_des"], frame_1_kp_des_filtered)
            #camera_1_good_matches, frame_1_old_pts_2d, frame_1_new_pts_2d = camera_1_2d_pts_result
            
            camera_2_lowe_matches = get_stereo_2d_correspondance(bf_matcher, prev_step["frame_2_kp_des"], frame_2_kp_des_filtered)
            #camera_2_good_matches, frame_2_old_pts_2d, frame_2_new_pts_2d = camera_2_2d_pts_result
            
            if len(camera_1_lowe_matches) == 0 and len(camera_2_lowe_matches) == 0:
                print("ignore failed to get camera 1 and 2 temporal matches")
                continue
            if len(camera_1_lowe_matches) == 0:
                print("ignore failed to get camera 1 temporal matches")
                continue
            if len(camera_2_lowe_matches) == 0:
                print("ignore failed to get camera 2 temporal matches")
                continue
            
            print("We got matches 222", len(camera_1_lowe_matches), len(camera_2_lowe_matches))

            # geometry check
            
            
            #print('len prev_step["frame_1_kp"]', len(prev_step["frame_1_kp"]))
            #print('len prev_step["frame_2_kp"]', len(prev_step["frame_2_kp"]))
            #print('len prev_step[frame_1_kp_des]', len(prev_step["frame_1_kp_des"]))
            #print('len prev_step[frame_2_kp_des]', len(prev_step["frame_2_kp_des"]))

            #print('prev_step["pts_3d"]', prev_step["pts_3d"].shape)
            #print('len frame_1_kp_filtered', len(frame_1_kp_filtered))
            #print('len frame_2_kp_filtered', len(frame_2_kp_filtered))
            #print('prev_step["stereo_matches"]', len(prev_step["stereo_matches"]))
            
            """
            len prev_step["frame_1_kp"] 89
            len prev_step["frame_2_kp"] 80
            len frame_1_kp_des 89
            len frame_2_kp_des 80
            prev_step["pts_3d"] (74, 3)
            len frame_1_kp_filtered 97
            len frame_2_kp_filtered 88
            Traceback (most recent call last):
            File "/home/jonathan/code/dronium-vision-2/stereo.pnp2.py", line 736, in <module>
                main()
            File "/home/jonathan/code/dronium-vision-2/stereo.pnp2.py", line 611, in main
                frame_1_old_pt_3d = prev_step["pts_3d"][old_idx]
            IndexError: index 74 is out of bounds for axis 0 with size 74
            """
            
            # now take the good matches for each of the cameras and recover the 3d map points and 2d points
            #for i,j in enumerate(prev_step["index_2d_des_3d"]["frame1"]):
            #    print("")
            
            camera_1_old_pts_3d = []
            camera_1_new_pts_2d = []
            for m in np.array(camera_1_lowe_matches):
                old_idx = m.queryIdx
                new_idx = m.trainIdx
                pt_3d_idx = prev_step["index_2d_des_3d"]["frame1"][old_idx]
                #print("pt_3d_idx", pt_3d_idx)
                frame_1_old_pt_3d = prev_step["pts_3d"][pt_3d_idx]
                frame_1_new_pt_2d = frame_1_kp_filtered[new_idx]
                camera_1_old_pts_3d.append(frame_1_old_pt_3d)
                camera_1_new_pts_2d.append(frame_1_new_pt_2d)
            camera_1_old_pts_3d = np.array(camera_1_old_pts_3d)
            camera_1_new_pts_2d = np.float32([kp.pt for kp in camera_1_new_pts_2d]).reshape(-1, 2)

            camera_2_old_pts_3d = []
            camera_2_new_pts_2d = []
            for m in np.array(camera_2_lowe_matches):
                old_idx = m.queryIdx
                new_idx = m.trainIdx
                #print("pt_3d_idx", pt_3d_idx)
                pt_3d_idx = prev_step["index_2d_des_3d"]["frame2"][old_idx]
                frame_2_old_pt_3d = prev_step["pts_3d"][pt_3d_idx]
                frame_2_new_pt_2d = frame_2_kp_filtered[new_idx]
                camera_2_old_pts_3d.append(frame_2_old_pt_3d)
                camera_2_new_pts_2d.append(frame_2_new_pt_2d)
            camera_2_old_pts_3d = np.array(camera_2_old_pts_3d)
            camera_2_new_pts_2d = np.float32([kp.pt for kp in camera_2_new_pts_2d]).reshape(-1, 2)


            # Use the new camera matrices obtained during rectification
            h1, w1 = frame1_img.shape[:2]
            h2, w2 = frame2_img.shape[:2]
            #undistorted_camera_1_matrix, _ = cv2.getOptimalNewCameraMatrix(K1, D1, (w1, h1), 1, (w1, h1))
            #undistorted_camera_2_matrix2, _ = cv2.getOptimalNewCameraMatrix(K2, D2, (w2, h2), 1, (w2, h2))

            print("3d points for camera 1, 2:", len(camera_1_old_pts_3d),len(camera_2_old_pts_3d))
            if (len(camera_1_old_pts_3d) < 70 or len(camera_2_old_pts_3d) < 70):
                print("ignore too few 3d points for camera 1, 2:", len(camera_1_old_pts_3d),len(camera_2_old_pts_3d))
                continue
            
            #print("camera_1_old_pts_3d", camera_1_old_pts_3d)
            #print("camera_2_old_pts_3d", camera_2_old_pts_3d)
            
            #print("camera_1_new_pts_2d", camera_1_new_pts_2d)
            #print("camera_2_new_pts_2d", camera_2_new_pts_2d)

            #frame_1_prior_rvec =  cv2.Rodrigues(prev_step["frame_1_R"])
            #frame_2_prior_rvec =  cv2.Rodrigues(prev_step["frame_2_R"])
            #frame_1_prior_tvec = prev_step["frame_1_T"][:3, 3]
            #frame_2_prior_tvec = prev_step["frame_2_T"][:3, 3]
            
            camera_1_ransac_results = cv2.solvePnPRansac(camera_1_old_pts_3d, camera_1_new_pts_2d, K1, D1, flags=cv2.SOLVEPNP_ITERATIVE, useExtrinsicGuess=True, rvec=cv2.Rodrigues(prev_step["frame_1_R"].copy())[0], tvec = prev_step["frame_1_T"].copy(), iterationsCount=3000, confidence=0.99999) #undistorted_camera_1_matrix, useExtrinsicGuess=True, rvec=frame_1_prior_rvec, tvec=frame_1_prior_tvec
            camera_1_pnp_ransac_success, R1_vec, T1_vec, camera_1_pnp_ransac_inliers_mask = camera_1_ransac_results
            #print("camera_1_ransac_results", camera_1_ransac_results)
            camera_2_ransac_results = cv2.solvePnPRansac(camera_2_old_pts_3d, camera_2_new_pts_2d, K2, D2, flags=cv2.SOLVEPNP_ITERATIVE, useExtrinsicGuess=True, rvec=cv2.Rodrigues(prev_step["frame_2_R"].copy())[0], tvec = prev_step["frame_2_T"].copy(), iterationsCount=3000, confidence=0.99999) #undistorted_camera_2_matrix2 useExtrinsicGuess=True, rvec=frame_2_prior_rvec, tvec=frame_2_prior_tvec
            #print("camera_2_ransac_results", camera_2_ransac_results)
            camera_2_pnp_ransac_success, R2_vec, T2_vec, camera_2_pnp_ransac_inliers_mask  = camera_2_ransac_results
            
            if not camera_1_pnp_ransac_success or not camera_2_pnp_ransac_success:
                continue
            
            #print("camera_1_pnp_ransac_success", camera_1_pnp_ransac_success)
            #print("R1_vec", R1_vec)
            #print("T1_vec", T1_vec)
            #print("R2_vec", R2_vec)
            #print("T2_vec", T2_vec)
            
            #R1, _ = cv2.Rodrigues(R1)
            #R2, _ = cv2.Rodrigues(R2)
            #print("R1", R1)
            #print("R2", R2)


            # Refine camera 1 pose using solvePnPRefineLM
            pnp_refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            camera_1_refine_result = cv2.solvePnPRefineLM(
                camera_1_old_pts_3d[camera_1_pnp_ransac_inliers_mask],
                camera_1_new_pts_2d[camera_1_pnp_ransac_inliers_mask],
                K1, #undistorted_camera_1_matrix,
                D1, #None,
                R1_vec,
                T1_vec,
                criteria=pnp_refine_criteria
            )
            #print ("camera_1_refine_result", camera_1_refine_result)
            R1_vec, T1_vec = camera_1_refine_result

            # Refine camera 2 pose using solvePnPRefineLM
            R2_vec, T2_vec = cv2.solvePnPRefineLM(
                camera_2_old_pts_3d[camera_2_pnp_ransac_inliers_mask],
                camera_2_new_pts_2d[camera_2_pnp_ransac_inliers_mask],
                K2, #undistorted_camera_2_matrix2,
                D2, # None,
                R2_vec,
                T2_vec,
                criteria=pnp_refine_criteria
            )
            
            """
            current_pos -= np.matrix(current_rot).T * np.matrix(tvec)
rmat, _ = cv.Rodrigues(rvec)
current_rot = rmat.dot(current_rot)
            """

            
            R1, _ = cv2.Rodrigues(R1_vec)
            R2, _ = cv2.Rodrigues(R2_vec)
            
            T1 = T1_vec
            T2 = T2_vec
            
            
            #T = np.hstack((R, tvec[:, np.newaxis]))
            #frame_1_R

            # Create overall pose
            P1 = np.hstack((R1, T1))
            P2 = np.hstack((R2, T2))
            
            #print("about to triangulate")
            print("R1", R1)
            print("T1", T1)
            print("R2", R2)
            print("T2", T2)
            
            #pts_3d_in_world_space = triangulate_points(P1,P2,inlier_frame_1_pts_2d.reshape(-1, 2).T, inlier_frame_2_pts_2d.reshape(-1, 2).T)
            pts_3d_in_camera_1_space = triangulate_points(K1, R1, T1, K2, R2, T2, inlier_frame_1_pts_2d.reshape(-1, 2).T, inlier_frame_2_pts_2d.reshape(-1, 2).T)
            pts_3d_in_world_space = transform_points_to_origin_2(pts_3d_in_camera_1_space, R1, T1) # pts_3d_in_camera_1_space# # here
            # fixme check that these points are in world space and do not need to be reoriented by the pose of camera 1
            
            #print("pts_3d_in_world_space", pts_3d_in_world_space.shape)
            #print("converted_matches", len(converted_matches))
            frame_1_kp_des_pt_3d_index = {}
            frame_2_kp_des_pt_3d_index = {}
            for idx_3d_pt, m in enumerate(converted_matches):
                exists = pts_3d_in_world_space[idx_3d_pt]
                old_idx = m.queryIdx
                frame_1_kp_des_pt_3d_index[old_idx] = idx_3d_pt
                new_idx = m.trainIdx
                frame_2_kp_des_pt_3d_index[new_idx] = idx_3d_pt
                
            step: StepType = {
                "frame1_img": frame1_img,
                "frame2_img": frame2_img,
                "frame_1_kp": frame_1_kp_filtered,
                "frame_2_kp": frame_2_kp_filtered,
                "frame_1_kp_des": frame_1_kp_des_filtered,
                "frame_2_kp_des": frame_2_kp_des_filtered,
                "frame_1_pts_2d": inlier_frame_1_pts_2d,
                "frame_2_pts_2d": inlier_frame_2_pts_2d,
                "frame_1_R": R1,
                "frame_1_T": T1,
                "frame_2_R": R2,
                "frame_2_T": T2,
                "frame_1_P": P1,
                "frame_2_P": P2,
                "pts_3d": pts_3d_in_world_space,
                "stereo_matches": converted_matches,
                "index_2d_des_3d": {
                    "frame1":frame_1_kp_des_pt_3d_index,
                    "frame2":frame_2_kp_des_pt_3d_index
                }
            }
            
            steps.append(step)
            
            visualize_all3(visualizer, calibration, camera_indicies, steps, wait_for_frame)
        
        cv2.imshow("stereo", final_img)    
        if k%256 == 27: # Escape
            break
    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
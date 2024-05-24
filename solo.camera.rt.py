from calib import detect_camera_indicies, load_camera_calibration
import cv2
import numpy as np
import time 
import open3d as o3d


def create_camera_frustum(camera_matrix, width, height, scale=1.0):
    fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]
    
    # Define the camera frustum vertices
    frustum_vertices = np.array([
        [0, 0, 0],
        [width, 0, 0],
        [width, height, 0],
        [0, height, 0],
        [width / 2, height / 2, -width / (2 * fx)],
    ])
    
    # Scale the frustum for better visualization
    frustum_vertices *= scale
    
    # Create lines connecting the frustum vertices
    frustum_lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [0, 4], [1, 4], [2, 4], [3, 4],
    ]
    
    return o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(frustum_vertices),
        lines=o3d.utility.Vector2iVector(frustum_lines)
    )


def get_orb_keypoint_and_descriptors(orb, frame):
    kp = orb.detect(frame,None)
    kp, des = orb.compute(frame, kp)
    return kp, des

def get_matcher():
    return cv2.BFMatcher(cv2.NORM_HAMMING) #, crossCheck=True

def get_covisible_features(matcher, des_current_frame, des_frame2):
    return matcher.knnMatch(des_current_frame,des_frame2, k=2)

def get_frame(cam):
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

def rotation_matrix_to_euler_angles(R):
  """
  Converts a rotation matrix to Euler angles using OpenCV's Rodrigues formula.

  Args:
    R: A 3x3 NumPy array representing the rotation matrix.

  Returns:
    A list of Euler angles (roll, pitch, yaw) in degrees.
  """

  trace = np.trace(R) - 1

  if trace > 0.999:  # Case 1: Singular - North pole
    yaw = 0
    pitch = np.pi / 2
    roll = np.arctan2(R[1, 2], R[0, 2])
  elif trace < -0.999:  # Case 2: Singular - South pole
    yaw = 0
    pitch = -np.pi / 2
    roll = np.arctan2(R[1, 2], R[0, 2])
  else:
    pitch = np.arccos(trace)
    sec_p = 1 / np.sin(pitch)
    yaw = np.arctan2(R[1, 0] * sec_p, R[0, 0] * sec_p)
    roll = np.arctan2(R[2, 1] * sec_p, R[2, 2] * sec_p)

  return np.degrees([roll, pitch, yaw])

# Function to combine poses
def combine_poses(from_pose, to_pose):
    combined_pose = {
        "R": np.dot(to_pose["R"], from_pose["R"]),
        "T": np.dot(to_pose["R"], from_pose["T"]) + to_pose["T"] # .reshape(-1, 1)
    }
    return combined_pose


#def combined_pose(initial_pose, R2, T2):
#    R_combined = np.dot(R2, initial_pose["R"])
#    T_combined = initial_pose["T"] + T2
#    return {"R": R_combined, "T": T_combined}

#def combine_poses(from_pose, to_pose):
#    combined_pose = {
#        "R": to_pose["R"],
#        "T": np.dot(to_pose["R"], from_pose["T"].reshape(-1, 1)) + to_pose["T"]
#    }
#    return combined_pose

#def combine_poses(from_pose, to_pose):
#    combined_pose = {
#        "R": np.dot(to_pose["R"], from_pose["R"]),
#        "T": np.dot(to_pose["R"], from_pose["T"]) + to_pose["T"]
#    }
#    print("from_pose[R]:", from_pose["R"])
#    print("from_pose[T]:", from_pose["T"])
#    print("to_pose[R]:", to_pose["R"])
#    print("to_pose[T]:", to_pose["T"])
#    print("combined_pose[R]:", combined_pose["R"])
#    print("combined_pose[T]:", combined_pose["T"])
#    return combined_pose
#def normalize_rotation(matrix):
#    q, r = np.linalg.qr(matrix)
#    return q

#def combine_poses(from_pose, to_pose):
#    combined_pose = {
#        "R": normalize_rotation(np.dot(to_pose["R"], from_pose["R"])),
#        "T": (to_pose["R"] @ from_pose["T"] + to_pose["T"]).astype(np.float64)
#    }
#    return combined_pose

# Function to combine poses
#def combine_poses(from_pose, to_pose):
#    combined_pose = {
#        "R": np.dot(to_pose["R"], from_pose["R"]),
#        "T": np.dot(to_pose["R"], from_pose["T"]) + to_pose["T"]
#    }
#    return combined_pose
#def combine_poses(from_pose, to_pose):
#    # Convert rotation matrices to axis-angle representation
#    axis_angle_from = cv2.Rodrigues(from_pose["R"])[0]
#    axis_angle_to = cv2.Rodrigues(to_pose["R"])[0]#

#    # Combine rotations using axis-angle representation
#    combined_axis_angle = axis_angle_to + axis_angle_from
#
#    # Convert back to rotation matrix
#    combined_rotation_matrix, _ = cv2.Rodrigues(combined_axis_angle)

#    # Combine translation vectors
#    combined_translation = np.dot(to_pose["R"], from_pose["T"]) + to_pose["T"]

#    combined_pose = {
#        "R": combined_rotation_matrix,
#        "T": combined_translation
#    }

#    return combined_pose

def compose_poses(initial_pose, delta_pose):
    """
    Composes an initial pose with a delta pose.

    Args:
        initial_pose: A dictionary containing the initial rotation ("R") and translation ("T").
        delta_pose: A dictionary containing the delta rotation ("dR") and translation ("dT").

    Returns:
        A dictionary containing the combined rotation ("R_combined") and translation ("T_combined").
    """

    R_initial = initial_pose["R"]
    T_initial = initial_pose["T"]
    dR = delta_pose["dR"]
    dT = delta_pose["dT"]

    # Combine rotations, applying dR to R_initial first
    R_combined = np.dot(R_initial, dR)

    # Combine translations, applying R_combined to dT first then adding T_initial
    T_combined = R_combined @ dT + T_initial

    return {"R": R_combined, "T": T_combined}

def compose_poses(initial_pose, delta_pose):
    """
    Composes two poses, considering the initial pose and a delta pose.

    Args:
        initial_pose: A dictionary containing the initial rotation ("R") and translation ("T").
        delta_pose: A dictionary containing the delta rotation ("dR") and translation ("dT").

    Returns:
        A dictionary containing the combined rotation ("R_combined") and translation ("T_combined").
    """

    R_initial = initial_pose["R"]
    T_initial = initial_pose["T"]
    dR = delta_pose["dR"]
    dT = delta_pose["dT"]

    # Combine rotations, applying dR to R_initial first
    R_combined = np.dot(R_initial, dR)

    # Combine translations, applying R_combined to dT first then adding T_initial
    T_combined = R_combined @ dT + T_initial

    return {"R": R_combined, "T": T_combined}

def project_3d_points_to_image(points_3d, camera_matrix, R, T):
    # Convert 3D points to homogeneous coordinates
    points_4d_homogeneous = cv2.convertPointsToHomogeneous(points_3d)
    
    # Apply the camera pose transformation
    points_3d_transformed = cv2.transform(points_4d_homogeneous, np.hstack((R, T)))
    
    # Convert back to non-homogeneous coordinates
    points_2d = cv2.convertPointsFromHomogeneous(points_3d_transformed).reshape(-1, 2)
    
    # Project the 3D points to image coordinates using the camera matrix
    points_2d_projected = cv2.projectPoints(points_3d, (0, 0, 0), (0, 0, 0), camera_matrix, None)[0].reshape(-1, 2)
    
    return points_2d, points_2d_projected

def calculate_translation_rotation_change(old_absolute_pose, new_absolute_pose):
    delta_rotation = np.dot(new_absolute_pose["R"], old_absolute_pose["R"].T)
    delta_translation = new_absolute_pose["T"] - old_absolute_pose["T"]

    # Compute the angular change for rotation
    angular_change = np.linalg.norm(cv2.Rodrigues(delta_rotation)[0])

    # Compute the Euclidean distance for translation
    translation_change = np.linalg.norm(delta_translation)

    # Normalize the changes if needed
    # For example, normalize by the magnitude of the cumulative rotation and translation
    normalization_factor_rotation = np.linalg.norm(old_absolute_pose["R"])
    normalization_factor_translation = np.linalg.norm(old_absolute_pose["T"])
    normalized_angular_change = angular_change / normalization_factor_rotation
    normalized_translation_change = translation_change# / normalization_factor_translation
    
    return normalized_angular_change, normalized_translation_change

def main():
    steps = []
    calibration = load_camera_calibration()
    
    camera_indicies = []
    for camera_index, _ in calibration.items():
        camera_indicies.append(camera_index)
        
    cam1 = cv2.VideoCapture(camera_indicies[0])

    n_features = 1000
    factor = 1.2
    orb = cv2.ORB_create(n_features, factor)
    bf_matcher = get_matcher()
    prev_time = time.time()
    last_frame = None
    
    # Create an Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Initialize cumulative pose
    cumulative_pose = {"R": np.eye(3), "T": np.zeros(3).reshape(3,1)}
    
    while True:
        current_frame = rectify_image(get_frame(cam1), calibration[camera_indicies[0]])

        if current_frame is None:
            #print("1")
            continue

        if last_frame is None:
            #print("2")
            last_frame = current_frame
            continue

        kp_frame_1, des_frame_1 = get_orb_keypoint_and_descriptors(orb, current_frame)
        kp_frame_2, des_frame_2 = get_orb_keypoint_and_descriptors(orb, last_frame)
        
        #print("des_frame_1", des_frame_1)
        
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
        src_pts:np.ndarray[(Ellipsis, 2), np.float32] = np.float32([kp_frame_1[m[0].queryIdx].pt for m in good_points]).reshape(-1, 2)
        dst_pts:np.ndarray[(Ellipsis, 2), np.float32] = np.float32([kp_frame_2[m[0].trainIdx].pt for m in good_points]).reshape(-1, 2)
        
        #print("dst_pts", dst_pts, dst_pts.shape, dst_pts.dtype)

        # Estimate fundamental matrix using RANSAC
        fundamental_result: tuple[cv2.typing.MatLike, cv2.typing.MatLike] = cv2.findFundamentalMat(src_pts, dst_pts, method=cv2.FM_RANSAC, ransacReprojThreshold=3.0, confidence=0.99, maxIters=2000)
        F, mask_fundamental = fundamental_result
        #] 
        final_img = cv2.drawMatchesKnn(current_frame,kp_frame_1,last_frame,kp_frame_2,good_points,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

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
            final_img = cv2.line(final_img, pt1, (pt2[0] + current_frame.shape[1], pt2[1]), (0, 255, 0), 2)

        # Draw lines connecting outliers on both images
        for m in np.array(good_points)[outlier_mask]:
            pt1 = tuple(map(int, kp_frame_1[m[0].queryIdx].pt))
            pt2 = tuple(map(int, kp_frame_2[m[0].trainIdx].pt))
            final_img = cv2.line(final_img, pt1, (pt2[0] + current_frame.shape[1], pt2[1]), (0, 0, 255), 2)
            
        final_good_points = np.array(good_points)[inlier_mask]
        #print("final_good_points", final_good_points, type(final_good_points), len(final_good_points))
        
        #for the good points we can triangulate
        # recover pose for 2nd camera
        try:
            retval, R2, T2, mask = cv2.recoverPose(F, src_pts[inlier_mask], dst_pts[inlier_mask])
        except:
            continue
        
        if retval < 30:
            print("ignore bad points", retval)
            #if retval < 2:
            #    print("rest last frame")
            #    last_frame = None
            continue

        triangulated_points_3d = triangulate_points(calibration[camera_indicies[0]]["camera_matrix"], cumulative_pose["R"], cumulative_pose["T"], calibration[camera_indicies[0]]["camera_matrix"], R2, T2, src_pts[inlier_mask].reshape(-1, 2).T, dst_pts[inlier_mask].reshape(-1, 2).T)

        # After triangulating points

        # Project 3D points into both camera views
        projected_pts_cam1, projected_pts_cam2 = project_3d_points_to_image(
            triangulated_points_3d, calibration[camera_indicies[0]]["camera_matrix"], cumulative_pose["R"], cumulative_pose["T"]
        )

        # Calculate the reprojection error
        reprojection_error_cam1 = np.mean(np.linalg.norm(src_pts[inlier_mask] - projected_pts_cam1, axis=1))
        reprojection_error_cam2 = np.mean(np.linalg.norm(dst_pts[inlier_mask] - projected_pts_cam2, axis=1))


        # Normalize the errors based on image dimensions or some other scale
        image_width = current_frame.shape[1]  # Replace with the actual image width
        image_height = current_frame.shape[0]  # Replace with the actual image height

        normalized_error_cam1 = reprojection_error_cam1 / np.sqrt(image_width * image_height)
        normalized_error_cam2 = reprojection_error_cam2 / np.sqrt(image_width * image_height)

        print("Normalized Reprojection Error Camera 1:", normalized_error_cam1)
        print("Normalized Reprojection Error Camera 2:", normalized_error_cam2)

        if normalized_error_cam1 > 2 or normalized_error_cam2 > 2:
            #print("ignore")
            #last_frame = None
            continue

        #if normalized_angular_change > 0.5:
        #    print("ignore")
        #    continue
        
        current_time = time.time()
        elapsed_time = current_time - prev_time
        fps = 1.0 / elapsed_time
        prev_time = current_time

        # Display FPS on the frame
        cv2.putText(final_img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        print("Score", retval)
        print("LASTC R", cumulative_pose["R"])
        print("LASTC T", cumulative_pose["T"])
        print("R2",R2)
        print("T2",T2)
        
        #cumulative_pose = {"R": R2, "T": cumulative_pose["T"] + T2}# combine_poses(cumulative_pose, ) #{"R": R2, "T": T2}# 
        old_pose = cumulative_pose.copy()
        new_pose = combine_poses({"R": R2, "T": T2}, cumulative_pose.copy())
        #cumulative_pose = {"R": R2, "T": cumulative_pose["T"] + T2}
        #cumulative_pose = {"R": R2, "T": np.matmul(R2, cumulative_pose["T"]) + T2} 
        #
        
        # combine_poses(cumulative_pose, ) #{"R": R2, "T": T2}#
        
        
        print("delta for old", calculate_translation_rotation_change({"R": np.eye(3), "T": np.zeros(3).reshape(3,1)}, new_pose))
        print("delta for new", calculate_translation_rotation_change(old_pose, new_pose))
        print("NEXTC R", new_pose["R"])
        print("NEXTC T", new_pose["T"])
        print("Eular angles", rotation_matrix_to_euler_angles(new_pose["R"]))


        cumulative_pose = new_pose
        
        # Create a camera frustum geometry
        camera_frustum = create_camera_frustum(
            calibration[camera_indicies[0]]["camera_matrix"],
            width=current_frame.shape[1],
            height=current_frame.shape[0],
            scale=0.1  # Adjust the scale for better visibility
        )
        
        # Update Open3D visualizer
        vis.clear_geometries()

        # Combine rotation matrix and translation vector into a 4x4 transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = cumulative_pose["R"]
        transformation_matrix[:3, 3] = cumulative_pose["T"][:, -1].flatten()

        # Add the camera frustum to the visualizer with the combined transformation matrix
        vis.add_geometry(camera_frustum.transform(transformation_matrix))

        vis.poll_events()
        vis.update_renderer()
        
        last_frame = current_frame
   
        cv2.imshow("stereo", final_img)    
        k = cv2.waitKey(1)
        if k%256 == 27: # Escape
            break
    cam1.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
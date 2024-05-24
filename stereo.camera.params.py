from calib import detect_camera_indicies, load_camera_calibration
import cv2
import numpy as np
import time 
import open3d as o3d


def visualize_point_cloud(points_3d):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3d)
    o3d.visualization.draw_geometries([point_cloud])
    
def visualize_all(all_steps):
    # Create an empty point cloud
    combined_points = np.empty((0, 3))

    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Loop through all steps and add 3D points and camera poses
    for idx, step in enumerate(all_steps):
        # Add 3D points
        combined_points = np.vstack((combined_points, step["3d_points"]))

        # Add camera pose for camera 1 with a red ball
        pose_cam1 = step["pose_frame_1"]["P"]
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=pose_cam1[:3, 3]))
        sphere_cam1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere_cam1.paint_uniform_color([1, 0, 0])  # Set the color to red

        sphere_cam1.translate(pose_cam1[:3, 3])  # Translate the sphere to the camera position
        vis.add_geometry(sphere_cam1)
        # Add camera pose for camera 2 with a blue ball
        pose_cam2 = step["pose_frame_2"]["P"]
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=pose_cam2[:3, 3]))
        sphere_cam2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere_cam2.paint_uniform_color([0, 1, 0])  # Set the color to blue

        sphere_cam2.translate(pose_cam2[:3, 3])  # Translate the sphere to the camera position
        vis.add_geometry(sphere_cam2)

    # Create a point cloud from all combined 3D points
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(combined_points)
    
    # Add the point cloud to the visualization
    vis.add_geometry(point_cloud)

    # Set the view control
    vis.get_view_control().set_front([0, 0, -1])
    vis.get_view_control().set_up([0, -1, 0])

    # Run the visualization
    vis.run()

    # Destroy the window
    vis.destroy_window()

def visualize_all3(calibration, camera_indicies, steps):
    # Step 0 - Init
    WIDTH = 1280
    HEIGHT = 720

    # Step 1 - Get scene objects
    #mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
    #sphere1.translate([0.1, 0.2, 0.1])

    # Step 2 - Create visualizer object
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.create_window(width=WIDTH, height=HEIGHT)

    # Step 3 - Add objects to visualizer
    visualizer.add_geometry(sphere1)
    #visualizer.add_geometry(mesh_frame)

    # Combine all 3D points from each step
    all_points = np.concatenate([step["3d_points"] for step in steps])

    # Visualize all 3D points
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(all_points)
    visualizer.add_geometry(point_cloud)

    for step in steps:
        
        # Step 4 - Get camera lines
        intrinsic_matrix = calibration[camera_indicies[0]]["camera_matrix"]
        extrinsic_matrix = step["pose_frame_1"]
        standardCameraParametersObj = visualizer.get_view_control().convert_to_pinhole_camera_parameters()
        
        # Extract rotation matrix ("R") and translation vector ("T") from the dictionary
        R = step["pose_frame_1"]["R"]
        T = step["pose_frame_1"]["T"]
        P = step["pose_frame_1"]["P"]

        # Create a 4x4 extrinsic matrix
        print("building custom matrix for extinsics", R, T, T.flatten())
        custom_extrinsic_matrix = np.eye(4)
        custom_extrinsic_matrix[:3, :3] = R
        custom_extrinsic_matrix[:3, 3] = T.flatten()
    
        print("standardCameraParametersObj.extrinsic", standardCameraParametersObj.extrinsic)
        print("extrinsic_matrix", extrinsic_matrix)
        print("custom_extrinsic_matrix", custom_extrinsic_matrix)
        cameraLines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=WIDTH, view_height_px=HEIGHT, intrinsic=intrinsic_matrix, extrinsic=custom_extrinsic_matrix)

        #visualizer.add_geometry(cameraLines)

        sphere_cam1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere_cam1.paint_uniform_color([1, 0, 0])  # Set the color to red
        sphere_cam1.translate(P[:3, 3])  # Translate the sphere to the camera position
        visualizer.add_geometry(sphere_cam1)

        # Create an arrow representing the camera orientation
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.2, cone_height=0.1
        )
        arrow.transform(custom_extrinsic_matrix)  # Apply the camera's extrinsic transformation
        visualizer.add_geometry(arrow)
        
        #### next

        # Step 4 - Get camera lines
        intrinsic_matrix = calibration[camera_indicies[1]]["camera_matrix"]
        extrinsic_matrix = step["pose_frame_2"]
        
        # Extract rotation matrix ("R") and translation vector ("T") from the dictionary
        R = step["pose_frame_2"]["R"]
        T = step["pose_frame_2"]["T"]
        P = step["pose_frame_2"]["P"]

        # Create a 4x4 extrinsic matrix
        print("building custom matrix for extinsics", R, T, T.flatten())
        custom_extrinsic_matrix = np.eye(4)
        custom_extrinsic_matrix[:3, :3] = R
        custom_extrinsic_matrix[:3, 3] = T.flatten()
    
        print("standardCameraParametersObj.extrinsic", standardCameraParametersObj.extrinsic)
        print("extrinsic_matrix", extrinsic_matrix)
        print("custom_extrinsic_matrix", custom_extrinsic_matrix)
        cameraLines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=WIDTH, view_height_px=HEIGHT, intrinsic=intrinsic_matrix, extrinsic=custom_extrinsic_matrix)

        visualizer.add_geometry(cameraLines)

        sphere_cam2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere_cam2.paint_uniform_color([0, 1, 0])  # Set the color to blue
        sphere_cam2.translate(P[:3, 3])  # Translate the sphere to the camera position
        visualizer.add_geometry(sphere_cam2)
        
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.2, cone_height=0.1
        )
        arrow.transform(custom_extrinsic_matrix)  # Apply the camera's extrinsic transformation
        visualizer.add_geometry(arrow)

    # Step 5 - Run visualizer
    visualizer.run()

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

def transform_to_world_coordinates(points_3d, camera_pose):
    # Extract rotation matrix ("R") and translation vector ("T") from the dictionary
    R = camera_pose["R"]
    T = camera_pose["T"]
    
    # Make sure the translation vector has the correct shape
    if len(T.shape) == 1:
        T = T.reshape((3, 1))
    
    # Apply the rotation and translation to the points
    transformed_points = np.dot(R, points_3d.T).T + T.flatten()
    
    return transformed_points

# Function to combine poses
def combine_poses(from_pose, to_pose):
    combined_pose = {
        "R": np.dot(to_pose["R"], from_pose["R"]),
        "T": np.dot(to_pose["R"], from_pose["T"]) + to_pose["T"]
    }
    return combined_pose

# Assuming the inverse function is defined
def inverse(pose):
    # Invert the rotation matrix
    inv_R = np.linalg.inv(pose["R"])
    # Negate the translation vector and rotate it back
    inv_T = -np.dot(inv_R, pose["T"])
    
    inverse_pose = {"R": inv_R, "T": inv_T}
    return inverse_pose

def next_pose_change(bf_matcher, kp_frame_1, des_frame_1, pose_frame_1, kp_frame_2, des_frame_2):
    try:
        matches = get_covisible_features(bf_matcher, des_frame_1, des_frame_2)
    except Exception as e:
        return None
        
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
    inlier_mask = mask_fundamental.ravel() == 1
    
    # recover pose for 2nd camera
    try:
        retval, R2, T2, mask = cv2.recoverPose(F, src_pts[inlier_mask], dst_pts[inlier_mask])
        
        # Get the old pose values
        R1 = pose_frame_1["R"]
        T1 = pose_frame_1["T"]
        
        # Combine the rotations and translations to get the new pose
        R_combined = np.dot(R2, R1)
        T_combined = T1 + T2
        
        # Return the new pose
        return {"R": R_combined, "T": T_combined, "P": np.hstack((R_combined, T_combined))}
    except:
        return None


def main():
    steps = []
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
        frame1 = rectify_image(get_frame(cam1), calibration[camera_indicies[0]])
        frame2 = rectify_image(get_frame(cam2), calibration[camera_indicies[1]])

        if frame1 is None or frame2 is None:
            continue

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
        
        #for the good points we can triangulate
        
        if len(steps) == 0:
            # initial
            # define camera 1 as being at zero,zero pose
            R1 = np.eye(3)
            T1 = np.zeros((3, 1))
            P1 = np.hstack((R1, T1))
            
            # recover pose for 2nd camera
            try:
                retval, R2, T2, mask = cv2.recoverPose(F, src_pts[inlier_mask], dst_pts[inlier_mask])
            except:
                continue
            
            P2 = np.hstack((R2, T2))
            
            # ok so now we have both cameras poses we need to get the 3d points triangulated into world space not
            # complete the code for this section 
            
            # Triangulate 3D points
            # 3D points are homogeneous, so divide by the last element to get non-homogeneous coordinates
            points_3d_in_world_space = triangulate_points(calibration[camera_indicies[0]]["camera_matrix"], R1, T1,
                                                        calibration[camera_indicies[1]]["camera_matrix"], R2, T2, src_pts[inlier_mask].reshape(-1, 2).T, dst_pts[inlier_mask].reshape(-1, 2).T)
            
            # todo fixme
            steps.append({"frame1": frame1, "kp_frame_1": kp_frame_1, "des_frame_1": des_frame_1, "pose_frame_1": {"R":R1, "T": T1, "P": P1}, "frame2": frame2, "kp_frame_2": kp_frame_2, "des_frame_2": des_frame_2,  "pose_frame_2": {"R":R2, "T": T2, "P": P2}, "3d_points": points_3d_in_world_space})
            visualize_all3(calibration, camera_indicies, steps)
            
        else:
            # deal with other situations... ignore for now
            prev_step = steps[len(steps)-1]
            
            camera1_old_pose = prev_step["pose_frame_1"]
            camera2_old_pose = prev_step["pose_frame_2"]
            
            # Recover pose for camera1 using camera2
            
            camera_2_old_pose_to_camera_1 = next_pose_change(bf_matcher, prev_step["kp_frame_2"], prev_step["des_frame_2"], camera2_old_pose, kp_frame_1, des_frame_1)


            # Recover pose for camera2 using camera1
            
            camera_1_old_pose_to_camera2 = next_pose_change(bf_matcher, prev_step["kp_frame_1"], prev_step["des_frame_1"], camera1_old_pose, kp_frame_2, des_frame_2)

            # Check if the new poses are successfully obtained
            if camera_2_old_pose_to_camera_1 is not None and camera_1_old_pose_to_camera2 is not None:
                # Assuming src_pts_new and dst_pts_new are the new corresponding inlier points obtained from the new frames
                src_pts_new = np.float32([kp_frame_1[m[0].queryIdx].pt for m in matches if m[0] in final_good_points]).reshape(-1, 2)
                dst_pts_new = np.float32([kp_frame_2[m[0].trainIdx].pt for m in matches if m[0] in final_good_points]).reshape(-1, 2)
                
                # now we need to combine the transformation such that we can go back to the origin.
                
                origin_to_camera_1_old_pose = camera1_old_pose
                origin_to_camera_2_old_pose = camera2_old_pose
                
                origin_to_camera_1 = (combine_poses(origin_to_camera_2_old_pose, camera_2_old_pose_to_camera_1)) #
                origin_to_camera_2 = (combine_poses(origin_to_camera_1_old_pose, camera_1_old_pose_to_camera2)) #
                camera_1_to_origin = inverse(origin_to_camera_1)
                camera_2_to_origin = inverse(origin_to_camera_2)
                
                # Triangulate 3D points with the new poses and inlier points
                points_3d_in_world_space = triangulate_points(calibration[camera_indicies[0]]["camera_matrix"],
                                                            camera_1_to_origin["R"], camera_1_to_origin["T"],
                                                            calibration[camera_indicies[1]]["camera_matrix"],
                                                            camera_2_to_origin["R"], camera_2_to_origin["T"],
                                                            src_pts_new.T, dst_pts_new.T)
                
                points_3d_in_world_space = transform_to_world_coordinates(points_3d_in_world_space, camera_1_to_origin)


                # Update the steps with new poses and 3D points
                steps.append({"frame1": frame1, "kp_frame_1": kp_frame_1, "des_frame_1": des_frame_1,
                            "pose_frame_1": camera_2_old_pose_to_camera_1, "frame2": frame2, "kp_frame_2": kp_frame_2,
                            "des_frame_2": des_frame_2, "pose_frame_2": camera_1_old_pose_to_camera2,
                            "3d_points": points_3d_in_world_space})
                
                visualize_all3(calibration, camera_indicies, steps)
        
        cv2.imshow("stereo", final_img)    
        k = cv2.waitKey(1)
        if k%256 == 27: # Escape
            break
    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
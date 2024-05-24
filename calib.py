import cv2
import numpy as np
import pickle
import os

def get_checker_board():
    return (9,6)

def get_world_coordinate_3d_points(checker_board):
    coords = np.zeros((1, checker_board[0] * checker_board[1], 3), np.float32)
    coords[0,:,:2] = np.mgrid[0:checker_board[0], 0:checker_board[1]].T.reshape(-1, 2)
    return coords 

def detect_camera_indicies(max_index_to_try=5):
    indicies = []
    for i in range(max_index_to_try):
        cam = cv2.VideoCapture(i)
        if cam.isOpened() is True:
            indicies.append(i)
        cam.release()
    return indicies

def checkerboard_calibration(camera_indicies, n_fits = 400):
    checker_board = get_checker_board()
    camera_parameters = {}
    calibration_img = cv2.imread("pattern.png")
    
    camera_calibration = {}
    counts = [0 for i in camera_indicies]
    cameras = [cv2.VideoCapture(camera_index) for camera_index in camera_indicies]
    
    print("Attempting to fit checkerboards ... move the checkerboard/camera around. Try to cover all angles and permutations.")

    cv2.imshow('Calibration pattern',calibration_img)
    k = cv2.waitKey(1)
        
    for camera_index, camera_identifier in enumerate(camera_indicies):
        cam = cameras[camera_index]
        world_points = []
        frame_points = []
        frame_grey = None     
        while True:
            # if already with enough data then continue with other camera
            if (counts[camera_index] == n_fits):
                cv2.destroyWindow('Camera %s calibration' % camera_index)
                break
            
            read_ret, frame = cam.read()
            if read_ret is False:
                print("Failed to read frame")
                continue
            
            frame_grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
            cv2.imshow('Camera %s calibration' % camera_index,frame_grey)
            k = cv2.waitKey(1)
            ret, corners = cv2.findChessboardCorners(frame_grey, checker_board, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret is True:
                world_points.append(get_world_coordinate_3d_points(checker_board))
                corners_sub_pix = cv2.cornerSubPix(frame_grey, corners, (11,11),(-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                frame_points.append(corners_sub_pix)
                counts[camera_index] += 1
                print("Detected a chessboard", counts)
                fit = cv2.drawChessboardCorners(frame_grey, checker_board, corners_sub_pix, ret)
                cv2.imshow('Camera %s calibration' % camera_index,fit)
                k = cv2.waitKey(1)
        camera_calibration[camera_identifier] = {"world_points": world_points, "frame_points": frame_points, "shape":frame_grey.shape[::-1]}
    
    for camera_index, camera_identifier in enumerate(camera_indicies):
        if all(count == n_fits for count in counts):
            cv2.destroyAllWindows()
            break        
    print("processing data... fitting camera parameters... please wait")
    
    for camera_identifier in camera_indicies:
        ret, camera_matrix, distortion_coeffs, R_vecs, T_vecs = cv2.calibrateCamera(camera_calibration[camera_identifier]["world_points"], camera_calibration[camera_identifier]["frame_points"], camera_calibration[camera_identifier]["shape"], None, None)
        camera_parameters[camera_identifier] = {"ret": ret, "camera_matrix": camera_matrix, "distortion_coeffs": distortion_coeffs, "R_vecs":R_vecs, "T_vecs":T_vecs}
        print("Computed parameters for camera", camera_identifier)

    print("Camera parameters found", camera_parameters)
        
    return camera_parameters              

def calibration_file_exists():
    return os.path.isfile("calibration.obj")

def save_camera_calibration(calibration_data):
    file = open("calibration.obj","wb")
    pickle.dump(calibration_data, file)
    file.close()

def load_camera_calibration():
    file = open("calibration.obj",'rb')
    calibration_data = pickle.load(file)
    file.close()
    return calibration_data

def rectify_image(frame, config):
    h, w = frame.shape[:2]
    camera_matrix =  config["camera_matrix"]
    distortion_coeffs  = config["distortion_coeffs"]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, (w, h), 1, (w, h))
    rectified_frame = cv2.undistort(frame, camera_matrix, distortion_coeffs, None, new_camera_matrix)
    return rectified_frame

def test_calibration(calibration):
    try:
        for i, config in calibration.items():
            cam = cv2.VideoCapture(i)
            ret, frame = cam.read()
            rectify_image(frame, config)
            cam.release()
            pp = (config["camera_matrix"][0, 2], config["camera_matrix"][1, 2])  # Principal point (cx, cy)
            focal = (config["camera_matrix"][0, 0], config["camera_matrix"][1, 1])  # Focal length (fx, fy)
            print("Principal Point (pp):", pp)
            print("Focal Length (focal):", focal)
        return True
    except Exception as e:
        print(e)
        return False

def auto_load_or_calibrate_cameras():
    calibration = None
    file_exists = calibration_file_exists()
    print("file_exists", file_exists)
    
    if calibration_file_exists() is True:
        print("calibration_file_exists")
        calibration = load_camera_calibration()
    calibration_working = False
    if calibration is not None:
        # check that calibration acutally works
        calibration_working = test_calibration(calibration)
        if calibration_working is True:
            return calibration
    # need a new calibration
    new_active_camera_indicies = detect_camera_indicies(5)
    calibration_data = checkerboard_calibration(new_active_camera_indicies)
    if len(calibration_data.keys()) != 0:
        save_camera_calibration(calibration_data)
    return load_camera_calibration()

def main():
    active_camera_indicies = detect_camera_indicies(5)
    print("active_camera_indicies", active_camera_indicies)
    calibration_data = checkerboard_calibration(active_camera_indicies)
    save_camera_calibration(calibration_data)
    print("Result")
    print(load_camera_calibration())

if __name__ == "__main__":
    main()
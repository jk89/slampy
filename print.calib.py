from calib import load_camera_calibration
camera_indicies = [0,2]
calibration = load_camera_calibration()

K1 = calibration[camera_indicies[0]]["camera_matrix"]
K2 = calibration[camera_indicies[1]]["camera_matrix"]
D1 = calibration[camera_indicies[0]]["distortion_coeffs"]
D2 = calibration[camera_indicies[1]]["distortion_coeffs"]

print(K1, K2)
print(D1, D2)
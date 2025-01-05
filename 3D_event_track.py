import cv2
import numpy as np
import json
import csv
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import os


# --- Configurable Parameters ---

CALIBRATION_FILE = '<path to your file here>'
LEFT_VIDEO_PATH = '<path to your file here'
RIGHT_VIDEO_PATH = '<path to your file here'
OUTPUT_LEFT_CSV = '<path to your file here'
OUTPUT_RIGHT_CSV = '<path to your file here'
OUTPUT_TRIANGULATION_CSV = '<path to your file here'  

# Display settings
SCALE_PERCENT = <Your Value>  # Percent of the original size for display windows
EPILINE_COLOR = <Your Value>  # Color for drawing epipolar lines (Green)
MATCH_LINE_COLOR = <Your Value>  # Color for drawing matching lines (Red)
BLOB_CIRCLE_COLOR = <Your Value>  # Color for drawing circles around matched blobs
BLOB_CIRCLE_RADIUS = <Your Value>  # Radius of circles drawn around matched blobs
BLOB_CIRCLE_THICKNESS = <Your Value>  # Thickness of circles around matched blobs (filled if negative)
MATCH_LINE_THICKNESS = <Your Value> # Thickness of the matching line between blobs

# Blob detection settings
BLOB_MEDIAN_BLUR_KERNEL = <Your Value>
BLOB_THRESHOLD_VALUE = <Your Value>
BLOB_MIN_SIZE = <Your Value> # Minimum size of detected blobs

# Kalman filter settings
KALMAN_PROCESS_NOISE_COV = <Your Value>  #example: np.array([[8, 0, 0, 0], [0, 8, 0, 0], [0, 0, 8, 0], [0, 0, 0, 8]], np.float32)
KALMAN_MEASUREMENT_NOISE_COV =<Your Value> #example: np.array([[1, 0], [0, 1]], np.float32)

KALMAN_ERROR_COV_PRE = <Your Value> #example:np.eye(4, dtype=np.float32) * 1000

# Gating thresholds for tracker association
MAX_VELOCITY_THRESHOLD = <Your Value>
VELOCITY_THRESHOLD = <Your Value>
MAX_DISTANCE_THRESHOLD = <Your Value>
DISTANCE_THRESHOLD = <Your Value>

# Tracker settings
MAX_LOST_FRAMES = <Your Value>  # Maximum allowed lost frames before a tracker is deactivated
COST_MATRIX_THRESHOLD = <Your Value>  # Threshold for cost matrix in linear assignment

FPS = <Your Value>  # Set this according to your video frame rate
TIME_PER_FRAME = 1 / FPS
INACTIVITY_THRESHOLD_FRAMES = 5 * FPS

# Display wait key settings
DISPLAY_WAIT_KEY = 1  # Milliseconds to wait between frames



# --------------------------------

# Global color map dictionary
color_map = {}

def get_color(id):
    """
    Retrieves the color associated with a given tracker ID.
    If the ID is not in the color_map, assigns a new random color.
    """
    if id not in color_map:
        np.random.seed(id)  # Seed with ID for consistent color assignment
        color_map[id] = tuple(np.random.randint(0, 255, size=3).tolist())
    return color_map[id]

# Load calibration data
def load_calibration_data(filename):
    with open(filename, 'r') as file:
        calib_data = json.load(file)
    return calib_data

def decompose_projection_matrix(P):
    K, R, t, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
    t = t[:3] / t[3]
    return K, R, t

def compute_fundamental_matrix(calib_data):
    P1 = np.array(calib_data['left_P'], dtype=np.float64)
    P2 = np.array(calib_data['right_P'], dtype=np.float64)
    K1, R1, t1 = decompose_projection_matrix(P1)
    K2, R2, t2 = decompose_projection_matrix(P2)
    R = R2 @ R1.T
    t = t2 - R @ t1
    E = np.cross(t.reshape(3), R)
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return F

def draw_epipolar_line(image, epiline):
    """Draw an epipolar line on the image."""
    h, w = image.shape[:2]
    if np.abs(epiline[1]) > 1e-5:
        x0, y0 = map(int, [0, -epiline[2] / epiline[1]])
        x1, y1 = map(int, [w, -(epiline[2] + epiline[0] * w) / epiline[1]])
        cv2.line(image, (x0, y0), (x1, y1), EPILINE_COLOR, 2)
    else:
        x0, x1 = 0, w
        y0 = int(-epiline[2] / (epiline[1] + 1e-5))
        y1 = int(-(epiline[2] + epiline[0] * w) / (epiline[1] + 1e-5))
        cv2.line(image, (x0, y0), (x1, y1), EPILINE_COLOR, 2)

def draw_matching_lines(left_image, right_image, matches):
    """Draw lines connecting matched points in the left and right images."""
    for match in matches:
        left_point = match['Left_Point']
        right_point = match['Right_Point']
        cv2.circle(left_image, left_point, BLOB_CIRCLE_RADIUS, BLOB_CIRCLE_COLOR, BLOB_CIRCLE_THICKNESS)
        cv2.circle(right_image, right_point, BLOB_CIRCLE_RADIUS, BLOB_CIRCLE_COLOR, BLOB_CIRCLE_THICKNESS)
        # Draw a horizontal line connecting the same y in the right image to show matches
        cv2.line(right_image, (right_point[0], right_point[1]),
                 (left_point[0], right_point[1]),
                 MATCH_LINE_COLOR, MATCH_LINE_THICKNESS)

def detect_blobs(frame):
    """Detect blobs in a given frame using contour detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, BLOB_MEDIAN_BLUR_KERNEL)
    _, binary = cv2.threshold(blurred, BLOB_THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(binary, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    contours, _ = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            size = cv2.contourArea(cnt)
            if size > BLOB_MIN_SIZE:
                blobs.append(((cx, cy), size, cnt))
    return blobs

def compute_epipolar_line(F, point):
    point_homogeneous = np.array([point[0], point[1], 1])
    epiline = F @ point_homogeneous
    return epiline

def calculate_point_to_line_distance(point, line):
    distance = np.abs(line[0] * point[0] + line[1] * point[1] + line[2]) / np.sqrt(line[0]**2 + line[1]**2)
    return distance

def compute_matching_scores(left_blobs, right_blobs, F):
    matches = []
    num_left_points = len(left_blobs)
    num_right_points = len(right_blobs)
    cost_matrix = np.zeros((num_left_points, num_right_points))

    for i, (left_point, _, _) in enumerate(left_blobs):
        epiline = compute_epipolar_line(F, left_point)

        for j, (right_point, _, _) in enumerate(right_blobs):
            distance = calculate_point_to_line_distance(right_point, epiline)
            cost_matrix[i, j] = distance

    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    for row, col in zip(row_indices, col_indices):
        if cost_matrix[row, col] < 9:
            match_info = {
                'Left_Point': left_blobs[row][0],
                'Right_Point': right_blobs[col][0],
                'Distance': cost_matrix[row, col]
            }
            matches.append(match_info)
    
    return matches

def triangulate_points(matches, calib_data):
    P1 = np.array(calib_data['left_P'], dtype=np.float64)
    P2 = np.array(calib_data['right_P'], dtype=np.float64)
    points_3D = []

    for match in matches:
        left_point_hom = np.array([match['Left_Point'][0], match['Left_Point'][1], 1], dtype=np.float64)
        right_point_hom = np.array([match['Right_Point'][0], match['Right_Point'][1], 1], dtype=np.float64)
        point_4D_hom = cv2.triangulatePoints(P1, P2, left_point_hom[:2], right_point_hom[:2])
        point_3D = point_4D_hom[:3] / point_4D_hom[3]
        points_3D.append(point_3D)

    points_3D = np.array(points_3D)  # Convert to numpy array for further processing

    if points_3D.ndim == 3:
        points_3D = points_3D.reshape(-1, 3)  # Flatten to 2D if it's 3D

    return points_3D

def calculate_reprojection_error(projection_matrix, points_3D, points_2D):
    # Ensure points_3D is a 2D array
    if points_3D.ndim == 3:
        points_3D = points_3D.reshape(-1, 3)

    # Convert 3D points to homogeneous coordinates
    points_3D_homogeneous = np.hstack((points_3D, np.ones((points_3D.shape[0], 1))))

    # Project 3D points back onto the 2D image plane
    projected_points_2D_homogeneous = projection_matrix @ points_3D_homogeneous.T

    # Convert homogeneous to Cartesian coordinates (2D)
    projected_points_2D = projected_points_2D_homogeneous[:2] / projected_points_2D_homogeneous[2]

    # Ensure projected_points_2D has the correct shape (N, 2)
    projected_points_2D = projected_points_2D.T  # Shape should be (N, 2) after transpose

    # Check that points_2D also has shape (N, 2)
    if points_2D.shape != projected_points_2D.shape:
        raise ValueError(f"Shape mismatch: points_2D has shape {points_2D.shape}, "
                         f"but projected_points_2D has shape {projected_points_2D.shape}.")

    # Compute reprojection error
    errors = np.linalg.norm(points_2D - projected_points_2D, axis=1)
    mean_error = np.mean(errors)

    return mean_error

class AdaptiveKalmanFilter:
    def __init__(self, initial_position):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], 
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], 
                                                 [0, 1, 0, 1], 
                                                 [0, 0, 1, 0], 
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = KALMAN_PROCESS_NOISE_COV
        self.kalman.measurementNoiseCov = KALMAN_MEASUREMENT_NOISE_COV
        self.kalman.errorCovPre = KALMAN_ERROR_COV_PRE
        self.kalman.statePre = np.array([[initial_position[0]], 
                                         [initial_position[1]], 
                                         [0], [0]], np.float32)
        self.kalman.statePost = np.array([[initial_position[0]], 
                                          [initial_position[1]], 
                                          [0], [0]], np.float32)
        self.adaptive_scale = 1.0

    def predict(self):
        return self.kalman.predict()

    def correct(self, measurement):
        predicted = self.kalman.predict()
        innovation = np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]]) - self.kalman.measurementMatrix.dot(predicted)
        innovation_magnitude = np.sqrt(innovation[0]**2 + innovation[1]**2)
        self.adapt_measurement_noise(innovation_magnitude)
        return self.kalman.correct(np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]], np.float32))

    def adapt_measurement_noise(self, innovation_magnitude):
        threshold_high = 10.0
        threshold_low = 2.0
        min_scale = 0.1
        max_scale = 10.0
        if innovation_magnitude > threshold_high:
            self.adaptive_scale = min(max_scale, self.adaptive_scale * 1.1)
        elif innovation_magnitude < threshold_low:
            self.adaptive_scale = max(min_scale, self.adaptive_scale * 0.9)
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * (1.0 * self.adaptive_scale)

class Tracker:
    def __init__(self, id, initial_position_left=None, initial_position_right=None, initial_size=0):
        self.id = id
        self.color = get_color(self.id)
        self.kalman_filter = AdaptiveKalmanFilter(initial_position_left if initial_position_left else (0,0))
        self.lost_frames = 0
        self.active = True
        self.vel_x = 0
        self.vel_y = 0
        self.vel_z = 0
        self.size = initial_size
        self.z_positions = []

        # Separate position lists for left and right views
        self.positions_left = []
        self.positions_right = []

        # If initial positions are known
        if initial_position_left:
            self.positions_left.append(initial_position_left)
        if initial_position_right:
            self.positions_right.append(initial_position_right)

        self.last_position = initial_position_left if initial_position_left else (0,0)
        self._3D_position = None
        self.last_matched_flag = 0

        print(f"Created Tracker ID: {self.id} with Color: {self.color}")

    def update_position(self, new_position_left=None, new_position_right=None, new_size=0, new_z=0):
        # Update velocities based on the left position if available
        prev_position = self.last_position
        if new_position_left:
            self.vel_x = (new_position_left[0] - prev_position[0]) / TIME_PER_FRAME
            self.vel_y = (new_position_left[1] - prev_position[1]) / TIME_PER_FRAME
            self.last_position = new_position_left
            self.positions_left.append(new_position_left)

        # If we have right position for a matched detection
        if new_position_right:
            self.positions_right.append(new_position_right)

        self.z_positions.append(new_z)
        self.size = new_size

    def set_3D_position(self, pos_3D):
        self._3D_position = pos_3D

    def has_3D_info(self):
        return self._3D_position is not None

    def get_3D_position(self):
        return self._3D_position if self._3D_position else (None, None, None)

    def size_of_blob(self):
        return self.size

    def get_velocity(self):
        return self.vel_x, self.vel_y, self.vel_z

    def predict_position(self):
        predicted_position = self.kalman_filter.predict()[:2]
        return (int(predicted_position[0]), int(predicted_position[1]))


class TrackerManager:
    def __init__(self, csv_file_path, triangulation_csv_path):
        self.trackers = []
        self.id_counter = 1
        self.max_lost_frames = MAX_LOST_FRAMES
        self.csv_file_path = csv_file_path
        self.triangulation_csv_path = triangulation_csv_path
        self.data_logged = False
        self.init_csv()

    def init_csv(self):
        with open(self.csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['ID', 'Frame', 'X', 'Y', 'Size', 'Vel_X', 'Vel_Y', 'Matched'])
        with open(self.triangulation_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['ID', 'Frame', 'X', 'Y', 'Z', 'Size', 'Vel_X', 'Vel_Y', 'Vel_Z'])

    def log_to_csv(self, tracker, frame_no, matched):
        if len(tracker.positions_left) > 0:
            last_position = tracker.positions_left[-1]
        elif len(tracker.positions_right) > 0:
            last_position = tracker.positions_right[-1]
        else:
            # If no positions at all, we cannot log this tracker
            return
    
        with open(self.csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                tracker.id,
                frame_no,
                last_position[0],
                last_position[1],
                tracker.size_of_blob(),
                tracker.vel_x,
                tracker.vel_y,
                matched
            ])
            self.data_logged = True
    
    def log_triangulation_to_csv(self, points_3D, frame_no, sizes, velocities, ids):
        if len(points_3D) > 0:
            with open(self.triangulation_csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                for idx, point_3D in enumerate(points_3D):
                    writer.writerow([
                        ids[idx], frame_no, point_3D[0], point_3D[1], point_3D[2],
                        sizes[idx], velocities[idx][0], velocities[idx][1], velocities[idx][2]
                    ])
                self.data_logged = True

    def update_trackers(self, detections, frame_no):
        # If no detections, increment lost_frames for all trackers
        if not detections:
            for tracker in self.trackers:
                tracker.lost_frames += 1
            # Remove trackers that have exceeded max_lost_frames
            self.trackers = [t for t in self.trackers if t.lost_frames < self.max_lost_frames]
            return

        detection_positions = []
        for det in detections:
            x_left, y_left, size, matched_flag, X_3D, Y_3D, Z_3D, x_right, y_right = det
            if x_left is not None and y_left is not None:
                detection_positions.append((x_left, y_left))
            elif x_right is not None and y_right is not None:
                detection_positions.append((x_right, y_right))
            else:
                # If neither left nor right is available (unexpected), skip
                detection_positions.append((0, 0))

        detection_positions = np.array(detection_positions)

        if self.trackers:
            # Get predicted positions from trackers
            predictions = np.array([t.predict_position() for t in self.trackers])

            # Compute cost matrix as Euclidean distance
            cost_matrix = np.linalg.norm(predictions[:, None] - detection_positions, axis=2)

            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            assigned_trackers = set()
            assigned_detections = set()

            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] < COST_MATRIX_THRESHOLD:
                    tracker = self.trackers[row]
                    # Extract detection data
                    x_left, y_left, size, matched_flag, X_3D, Y_3D, Z_3D, x_right, y_right = detections[col]

                    measure_x = x_left if x_left is not None else x_right
                    measure_y = y_left if y_left is not None else y_right

                    corrected_position = tracker.kalman_filter.correct((measure_x, measure_y))

                    tracker.update_position(
                        new_position_left=(x_left, y_left) if x_left is not None and y_left is not None else None,
                        new_position_right=(x_right, y_right) if x_right is not None and y_right is not None else None,
                        new_size=size
                    )
                    tracker.lost_frames = 0
                    tracker.active = True
                    tracker.last_matched_flag = matched_flag

                    if X_3D is not None and Y_3D is not None and Z_3D is not None:
                        tracker.set_3D_position((X_3D, Y_3D, Z_3D))

                    assigned_trackers.add(row)
                    assigned_detections.add(col)

            # Increment lost frames for unassigned trackers
            for i, tracker in enumerate(self.trackers):
                if i not in assigned_trackers:
                    tracker.lost_frames += 1
                    if tracker.lost_frames > self.max_lost_frames:
                        tracker.active = False

            # Create new trackers for unassigned detections
            for i, det in enumerate(detections):
                if i not in assigned_detections:
                    x_left, y_left, size, matched_flag, X_3D, Y_3D, Z_3D, x_right, y_right = det

                    init_left_pos = (x_left, y_left) if x_left is not None and y_left is not None else None
                    init_right_pos = (x_right, y_right) if x_right is not None and y_right is not None else None

                    if init_left_pos is None and init_right_pos is not None:
                        init_left_pos = init_right_pos
                    elif init_left_pos is None and init_right_pos is None:
                        continue

                    new_tracker = Tracker(
                        self.id_counter,
                        initial_position_left=init_left_pos,
                        initial_position_right=init_right_pos,
                        initial_size=size
                    )
                    new_tracker.last_matched_flag = matched_flag
                    if X_3D is not None and Y_3D is not None and Z_3D is not None:
                        new_tracker.set_3D_position((X_3D, Y_3D, Z_3D))
                    self.trackers.append(new_tracker)
                    self.id_counter += 1

            # Remove inactive trackers
            self.trackers = [t for t in self.trackers if t.active]

        else:
            # No existing trackers, create new ones for all detections
            for det in detections:
                x_left, y_left, size, matched_flag, X_3D, Y_3D, Z_3D, x_right, y_right = det

                init_left_pos = (x_left, y_left) if x_left is not None and y_left is not None else None
                init_right_pos = (x_right, y_right) if x_right is not None and y_right is not None else None

                if init_left_pos is None and init_right_pos is not None:
                    init_left_pos = init_right_pos
                elif init_left_pos is None and init_right_pos is None:
                    continue

                new_tracker = Tracker(
                    self.id_counter,
                    initial_position_left=init_left_pos,
                    initial_position_right=init_right_pos,
                    initial_size=size
                )
                new_tracker.last_matched_flag = matched_flag
                if X_3D is not None and Y_3D is not None and Z_3D is not None:
                    new_tracker.set_3D_position((X_3D, Y_3D, Z_3D))
                self.trackers.append(new_tracker)
                self.id_counter += 1

def main():
    # Load calibration data
    calib_data = load_calibration_data(CALIBRATION_FILE)
    F = compute_fundamental_matrix(calib_data)

    # Extract calibration parameters
    P1 = np.array(calib_data['left_P'], dtype=np.float64)
    P2 = np.array(calib_data['right_P'], dtype=np.float64)
    K1 = np.array(calib_data['left_camera_matrix'], dtype=np.float64)
    K2 = np.array(calib_data['right_camera_matrix'], dtype=np.float64)
    dist1 = np.array(calib_data['left_dist'], dtype=np.float64)
    dist2 = np.array(calib_data['right_dist'], dtype=np.float64)
    R = np.array(calib_data['R'], dtype=np.float64)
    T = np.array(calib_data['T'], dtype=np.float64)
    image_size = tuple(calib_data['DIM'])

    # Compute rectification and projection matrices for stereo setup
    R1, R2, P1_rect, P2_rect, Q, _, _ = cv2.stereoRectify(
        K1, dist1, K2, dist2, image_size, R, T
    )

    # Initialize rectification maps for both cameras
    map1_left, map2_left = cv2.initUndistortRectifyMap(
        K1, dist1, R1, P1_rect, image_size, cv2.CV_32FC1
    )
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        K2, dist2, R2, P2_rect, image_size, cv2.CV_32FC1
    )

    # Open video captures
    left_cap = cv2.VideoCapture(LEFT_VIDEO_PATH)
    right_cap = cv2.VideoCapture(RIGHT_VIDEO_PATH)

    if not left_cap.isOpened() or not right_cap.isOpened():
        print("Error: Could not open one of the video files.")
        return

    frame_width = int(left_cap.get(cv2.CAP_PROP_FRAME_WIDTH) * SCALE_PERCENT / 100)
    frame_height = int(left_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * SCALE_PERCENT / 100)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    
    # --- VIDEO WRITERS ---
    out_left = cv2.VideoWriter(
        'left_no_trajectory.mp4', 
        fourcc, 
        fps, 
        (frame_width, frame_height)
    )
    out_right_epilines = cv2.VideoWriter(
        'right_no_trajectory_epilines.mp4',
        fourcc, 
        fps, 
        (frame_width, frame_height)
    )
    out_left_detection = cv2.VideoWriter(
        '/output_left_detection_no_trajectory.mp4',
        fourcc, 
        fps, 
        (frame_width, frame_height)
    )
    out_right_detection = cv2.VideoWriter(
        'output_right_detection_no_trajectory.mp4',
        fourcc, 
        fps, 
        (frame_width, frame_height)
    )

    manager = TrackerManager(OUTPUT_LEFT_CSV, OUTPUT_TRIANGULATION_CSV)

    frame_no = 0
    total_reprojection_error_left = 0
    total_reprojection_error_right = 0
    total_frames = 0

    total_left_blobs_detected = 0
    total_right_blobs_detected = 0
    total_blobs_matched = 0

    while True:
        ret_left, left_frame = left_cap.read()
        ret_right, right_frame = right_cap.read()

        if not ret_left or not ret_right:
            print("End of one of the video streams or cannot read the frame.")
            break

        # Apply stereo rectification
        rectified_left = cv2.remap(left_frame, map1_left, map2_left, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(right_frame, map1_right, map2_right, cv2.INTER_LINEAR)

        left_detection_frame = rectified_left.copy()
        right_detection_frame = rectified_right.copy()

        # Detect blobs
        left_blobs = detect_blobs(rectified_left)
        right_blobs = detect_blobs(rectified_right)

        total_left_blobs_detected += len(left_blobs)
        total_right_blobs_detected += len(right_blobs)

        # Draw detection circles on copies
        for blob in left_blobs:
            cv2.circle(left_detection_frame, blob[0], BLOB_CIRCLE_RADIUS, (255, 0, 0), 2)
        for blob in right_blobs:
            cv2.circle(right_detection_frame, blob[0], BLOB_CIRCLE_RADIUS, (255, 0, 0), 2)

        # Compute matches
        matches = compute_matching_scores(left_blobs, right_blobs, F)
        total_blobs_matched += len(matches)

        # Triangulate matched points
        points_3D = []
        if len(matches) > 0:
            points_3D = triangulate_points(matches, calib_data)

        # Identify unmatched blobs on left
        unmatched_left = [
            (blob[0], blob[1]) for blob in left_blobs
            if blob[0] not in [m['Left_Point'] for m in matches]
        ]

        # Identify unmatched blobs on right
        unmatched_right = [
            (blob[0], blob[1]) for blob in right_blobs
            if blob[0] not in [m['Right_Point'] for m in matches]
        ]

        # Build unified detection list
        unified_detections = []

        # For matched pairs: create one detection
        for i, match in enumerate(matches):
            left_pt = match['Left_Point']
            right_pt = match['Right_Point']
            size_left = next(b[1] for b in left_blobs if b[0] == left_pt)
            size_right = next(b[1] for b in right_blobs if b[0] == right_pt)
            X_3D, Y_3D, Z_3D = (None, None, None)
            if points_3D is not None and len(points_3D) > i:
                X_3D, Y_3D, Z_3D = points_3D[i]

            unified_detections.append(
                (left_pt[0], left_pt[1],
                 size_left, 1,
                 X_3D, Y_3D, Z_3D,
                 right_pt[0], right_pt[1])
            )

        # For unmatched left:
        for (pt, sz) in unmatched_left:
            unified_detections.append((pt[0], pt[1], sz, 0, None, None, None, None, None))

        # For unmatched right:
        for (pt, sz) in unmatched_right:
            unified_detections.append((None, None, sz, 0, None, None, None, pt[0], pt[1]))

        # Update trackers
        manager.update_trackers(unified_detections, frame_no)

        # Log trackers
        for tracker in manager.trackers:
            manager.log_to_csv(tracker, frame_no, matched=tracker.last_matched_flag)
            if tracker.has_3D_info():
                X_3D, Y_3D, Z_3D = tracker.get_3D_position()
                vx, vy, vz = tracker.get_velocity()
                manager.log_triangulation_to_csv(
                    [(X_3D, Y_3D, Z_3D)],
                    frame_no,
                    [tracker.size],
                    [(vx, vy, vz)],
                    [tracker.id]
                )

        # Compute reprojection error if there were matched points
        if len(matches) > 0 and len(points_3D) == len(matches):
            left_points_2D = np.array([m['Left_Point'] for m in matches])
            right_points_2D = np.array([m['Right_Point'] for m in matches])
            reprojection_error_left = calculate_reprojection_error(
                calib_data['left_P'], points_3D, left_points_2D
            )
            reprojection_error_right = calculate_reprojection_error(
                calib_data['right_P'], points_3D, right_points_2D
            )
            total_reprojection_error_left += reprojection_error_left
            total_reprojection_error_right += reprojection_error_right
            total_frames += 1

        # Draw epipolar lines for matched detections on the right frame
        for det in unified_detections:
            if det[3] == 1:  # matched
                left_pt = (int(det[0]), int(det[1]))
                epiline = compute_epipolar_line(F, left_pt)
                draw_epipolar_line(rectified_right, epiline)

        # Draw matching lines (but NOT trajectories) for visualization
        draw_matching_lines(rectified_left, rectified_right, matches)

        # Resize frames
        dim = (frame_width, frame_height)
        resized_left_frame = cv2.resize(rectified_left, dim, interpolation=cv2.INTER_AREA)
        resized_right_frame = cv2.resize(rectified_right, dim, interpolation=cv2.INTER_AREA)
        resized_left_detection_frame = cv2.resize(left_detection_frame, dim, interpolation=cv2.INTER_AREA)
        resized_right_detection_frame = cv2.resize(right_detection_frame, dim, interpolation=cv2.INTER_AREA)

        # Write frames to files (no trajectories drawn)
        out_left.write(resized_left_frame)
        out_right_epilines.write(resized_right_frame)
        out_left_detection.write(resized_left_detection_frame)
        out_right_detection.write(resized_right_detection_frame)

        # Display
        cv2.imshow('Left Frame (No Trajectories)', resized_left_frame)
        cv2.imshow('Right Frame with Epipolar Lines (No Trajectories)', resized_right_frame)
        cv2.imshow('Left Frame Detections (No Trajectories)', resized_left_detection_frame)
        cv2.imshow('Right Frame Detections (No Trajectories)', resized_right_detection_frame)

        if cv2.waitKey(DISPLAY_WAIT_KEY) & 0xFF == ord('q'):
            print("Quitting the video processing loop.")
            break

        frame_no += 1

    # Cleanup
    left_cap.release()
    right_cap.release()
    out_left.release()
    out_right_epilines.release()
    out_left_detection.release()
    out_right_detection.release()
    cv2.destroyAllWindows()

    # Print statistics
    print(f"Total Left Blobs Detected: {total_left_blobs_detected}")
    print(f"Total Right Blobs Detected: {total_right_blobs_detected}")
    print(f"Total Matched Blobs (Pairs): {total_blobs_matched}")

    if total_frames > 0:
        overall_reprojection_error_left = total_reprojection_error_left / total_frames
        overall_reprojection_error_right = total_reprojection_error_right / total_frames
        print(f"Overall Reprojection Error - Left: {overall_reprojection_error_left:.4f}, "
              f"Right: {overall_reprojection_error_right:.4f}")

if __name__ == '__main__':
    main()

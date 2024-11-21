import cv2
import numpy as np
import json
import csv
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import os


# --- Configurable Parameters ---

CALIBRATION_FILE = '/home/pratham/UNSW/camera_calib/stereo-camera-calibration/calibration_results_new_4.10/calibration_results.json'
LEFT_VIDEO_PATH = '/home/pratham/UNSW/camera_calib/stereo-camera-calibration/bee_flights_2-10/rec3/left.avi'
RIGHT_VIDEO_PATH = '/home/pratham/UNSW/camera_calib/stereo-camera-calibration/bee_flights_2-10/rec3/right.avi'
OUTPUT_LEFT_CSV = '/home/pratham/UNSW/camera_calib/stereo-camera-calibration/plots/vicon/exp4/output_left.csv'
OUTPUT_RIGHT_CSV = '/home/pratham/UNSW/camera_calib/stereo-camera-calibration/plots/vicon/exp4/output_right.csv'
OUTPUT_3D_CSV = '/home/pratham/UNSW/camera_calib/stereo-camera-calibration/plots/vicon/exp4/output_3D.csv'
OUTPUT_TRIANGULATION_CSV = '/home/pratham/UNSW/camera_calib/stereo-camera-calibration/plots/vicon/exp4/output_triangulationcheck.csv'  # New CSV for 3D points

# Display settings
SCALE_PERCENT = 75  # Percent of the original size for display windows
EPILINE_COLOR = (0, 255, 0)  # Color for drawing epipolar lines (Green)
MATCH_LINE_COLOR = (0, 0, 255)  # Color for drawing matching lines (Red)
BLOB_CIRCLE_COLOR = (0, 0, 255)  # Color for drawing circles around matched blobs
BLOB_CIRCLE_RADIUS = 10  # Radius of circles drawn around matched blobs
BLOB_CIRCLE_THICKNESS = -1  # Thickness of circles around matched blobs (filled if negative)
MATCH_LINE_THICKNESS = 1  # Thickness of the matching line between blobs

# Blob detection settings
BLOB_MEDIAN_BLUR_KERNEL = 5
BLOB_THRESHOLD_VALUE = 70
BLOB_MIN_SIZE = 20  # Minimum size of detected blobs

# Kalman filter settings
KALMAN_PROCESS_NOISE_COV = np.array([[8, 0, 0, 0], [0, 8, 0, 0], [0, 0, 8, 0], [0, 0, 0, 8]], np.float32)
KALMAN_MEASUREMENT_NOISE_COV = np.array([[1, 0], [0, 1]], np.float32)

KALMAN_ERROR_COV_PRE = np.eye(4, dtype=np.float32) * 1000

# Gating thresholds for tracker association
MAX_VELOCITY_THRESHOLD = 30
VELOCITY_THRESHOLD = 30
MAX_DISTANCE_THRESHOLD = 5
DISTANCE_THRESHOLD = 5

# Tracker settings
MAX_LOST_FRAMES = 8  # Maximum allowed lost frames before a tracker is deactivated
COST_MATRIX_THRESHOLD = 180  # Threshold for cost matrix in linear assignment

FPS = 30  # Set this according to your video frame rate
TIME_PER_FRAME = 1 / FPS

# Display wait key settings
DISPLAY_WAIT_KEY = 1  # Milliseconds to wait between frames



# --------------------------------

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
        cv2.line(right_image, (right_point[0], right_point[1]), (left_point[0], right_point[1]), MATCH_LINE_COLOR, MATCH_LINE_THICKNESS)

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



def compute_matching_scores(left_blobs, right_blobs, F, left_trackers, right_trackers):
    """Compute matching scores considering epipolar distance, blob size, and velocity."""
    matches = []
    num_left_points = len(left_blobs)
    num_right_points = len(right_blobs)
    cost_matrix = np.full((num_left_points, num_right_points), np.inf)  # Initialize cost matrix with large values

    for i, (left_point, left_size, _) in enumerate(left_blobs):
        epiline = compute_epipolar_line(F, left_point)

        for j, (right_point, right_size, _) in enumerate(right_blobs):
            # Compute epipolar distance
            epipolar_distance = calculate_point_to_line_distance(right_point, epiline)

            # Compute size difference
            size_difference = abs(left_size - right_size)

            # Compute velocity cost if trackers exist for these blobs
            velocity_cost = 0
            left_tracker = find_tracker_by_position(left_trackers, left_point)
            right_tracker = find_tracker_by_position(right_trackers, right_point)
            if left_tracker and right_tracker:
                vel_left = np.array([left_tracker.vel_x, left_tracker.vel_y])
                vel_right = np.array([right_tracker.vel_x, right_tracker.vel_y])
                velocity_cost = np.linalg.norm(vel_left - vel_right)

            # Compute overall cost as weighted sum of components
            cost = (
                epipolar_distance * 0.5  # Weight for epipolar distance
                + size_difference * 0.3  # Weight for size difference
                + velocity_cost * 0.2  # Weight for velocity difference
            )

            cost_matrix[i, j] = cost

    # Solve the assignment problem
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    for row, col in zip(row_indices, col_indices):
        if cost_matrix[row, col] < COST_MATRIX_THRESHOLD:
            match_info = {
                'Left_Point': left_blobs[row][0],
                'Right_Point': right_blobs[col][0],
                'Distance': cost_matrix[row, col],
                'Size_Difference': abs(left_blobs[row][1] - right_blobs[col][1]),
                'Epipolar_Distance': calculate_point_to_line_distance(right_blobs[col][0], compute_epipolar_line(F, left_blobs[row][0]))
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
        
        # Invert Y-axis to make it increase upwards
        point_3D[1] = -point_3D[1]
        
        points_3D.append(point_3D)

    points_3D = np.array(points_3D)  # Convert to numpy array for further processing

    if points_3D.ndim == 3:
        points_3D = points_3D.reshape(-1, 3)  # Flatten to 2D if it's 3D

    return points_3D

def calculate_reprojection_error(rectified_projection_matrix, points_3D, rectified_points_2D):
    # Ensure points_3D is a 2D array (N, 3)
    if points_3D.ndim == 3:
        points_3D = points_3D.reshape(-1, 3)

    # Convert 3D points to homogeneous coordinates by adding an additional dimension of 1s
    points_3D_homogeneous = np.hstack((points_3D, np.ones((points_3D.shape[0], 1))))

    # Project 3D points onto the 2D rectified image plane using the rectified projection matrix
    projected_points_2D_homogeneous = rectified_projection_matrix @ points_3D_homogeneous.T

    # Convert homogeneous 2D coordinates to Cartesian (x, y) coordinates
    projected_points_2D = projected_points_2D_homogeneous[:2] / projected_points_2D_homogeneous[2]

    # Ensure projected_points_2D has the correct shape (N, 2) after transpose
    projected_points_2D = projected_points_2D.T  # Shape should be (N, 2) after transpose

    # Check that the shape of rectified_points_2D matches projected_points_2D
    if rectified_points_2D.shape != projected_points_2D.shape:
        raise ValueError(f"Shape mismatch: rectified_points_2D has shape {rectified_points_2D.shape}, "
                         f"but projected_points_2D has shape {projected_points_2D.shape}.")

    # Compute the reprojection error as the Euclidean distance between actual and projected 2D points
    errors = np.linalg.norm(rectified_points_2D - projected_points_2D, axis=1)

    # Compute the mean reprojection error
    mean_error = np.mean(errors)

    return mean_error


class AdaptiveKalmanFilter:
    def __init__(self, initial_position):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = KALMAN_PROCESS_NOISE_COV
        self.kalman.measurementNoiseCov = KALMAN_MEASUREMENT_NOISE_COV
        self.kalman.errorCovPre = KALMAN_ERROR_COV_PRE
        self.kalman.statePre = np.array([[initial_position[0]], [initial_position[1]], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[initial_position[0]], [initial_position[1]], [0], [0]], np.float32)
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
    def __init__(self, id, initial_position, initial_size, initial_z=0):
        self.id = id
        self.kalman_filter = AdaptiveKalmanFilter(initial_position)
        self.lost_frames = 0
        self.active = True
        self.color = self.generate_color()
        self.positions = [initial_position]
        self.last_position = initial_position
        self.vel_x = 0
        self.vel_y = 0
        self.vel_z = 0  # Initialize Z velocity
        self.size = initial_size  # Store the initial size of the blob
        self.z_positions = [initial_z]  # Track Z positions

    def generate_color(self):
        np.random.seed(self.id)
        return tuple(np.random.randint(0, 255, size=3).tolist())

    def update_position(self, new_position, new_size, new_z=0):
        if len(self.positions) > 0:
            prev_position = self.positions[-1]
            self.vel_x = (new_position[0] - prev_position[0]) / TIME_PER_FRAME
            self.vel_y = (new_position[1] - prev_position[1]) / TIME_PER_FRAME
            if len(self.z_positions) > 0:
                prev_z = self.z_positions[-1]
                self.vel_z = (new_z - prev_z) / TIME_PER_FRAME  # Calculate Z velocity
        self.positions.append(new_position)
        self.z_positions.append(new_z)  # Store the new Z position
        self.last_position = new_position
        self.size = new_size  # Update the size of the blob

    def size_of_blob(self):
        return self.size  # Return the size of the blob

    def predict_position(self):
        predicted_position = self.kalman_filter.predict()[:2]
        return (int(predicted_position[0]), int(predicted_position[1]))

    def get_velocity(self):
        return self.vel_x, self.vel_y, self.vel_z  # Return X, Y, Z velocities
        
    def get_smoothed_positions(self, window_size=5):
        if len(self.positions) < window_size:
            return self.positions

        smoothed_positions = []
        for i in range(len(self.positions)):
            start_index = max(0, i - window_size + 1)
            end_index = i + 1
            window_positions = self.positions[start_index:end_index]
            avg_x = sum(pos[0] for pos in window_positions) / len(window_positions)
            avg_y = sum(pos[1] for pos in window_positions) / len(window_positions)
            smoothed_positions.append((avg_x, avg_y))
        return smoothed_positions


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
        if len(tracker.positions) > 0:
            # Check velocity against the threshold
            vel_x, vel_y, _ = tracker.get_velocity()
            velocity_magnitude = np.sqrt(vel_x**2 + vel_y**2)

            # If the velocity exceeds the threshold, skip logging this data point
            if velocity_magnitude > MAX_VELOCITY_THRESHOLD:
                print(f"2D Tracker {tracker.id}: Point skipped due to high velocity ({velocity_magnitude:.2f}).")
                return

            with open(self.csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                last_position = tracker.positions[-1]
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
                    vel_x, vel_y, vel_z = velocities[idx]
                    velocity_magnitude = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)

                    # Filter out points based on velocity and Z distance
                    if velocity_magnitude > MAX_VELOCITY_THRESHOLD or np.abs(point_3D[2]) > MAX_DISTANCE_THRESHOLD:
                        print(f"3D Tracker {ids[idx]}: Point skipped due to outlier ({velocity_magnitude:.2f} velocity or Z-distance).")
                        continue
                    
                    writer.writerow([
                        ids[idx], frame_no, point_3D[0], point_3D[1], point_3D[2], 
                        sizes[idx], vel_x, vel_y, vel_z
                    ])
                self.data_logged = True

    def update_trackers(self, detections, frame_no):
        if not detections:
            for tracker in self.trackers:
                tracker.lost_frames += 1
            self.trackers = [tracker for tracker in self.trackers if tracker.lost_frames < self.max_lost_frames]
            return

        if not self.trackers:
            for det, size in detections:
                new_tracker = Tracker(self.id_counter, det, size)
                self.trackers.append(new_tracker)
                self.id_counter += 1
            return

        predictions = np.array([t.kalman_filter.predict()[:2].flatten() for t in self.trackers])
        detections_np = np.array([det[0] for det in detections])

        cost_matrix = np.linalg.norm(predictions[:, None] - detections_np, axis=2)
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        assigned_trackers = set()
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < COST_MATRIX_THRESHOLD:
                corrected_position = self.trackers[row].kalman_filter.correct(detections[col][0])
                self.trackers[row].update_position((corrected_position[0][0], corrected_position[1][0]), detections[col][1])
                self.trackers[row].lost_frames = 0
                self.trackers[row].active = True
                assigned_trackers.add(row)

        for i, tracker in enumerate(self.trackers):
            if i not in assigned_trackers:
                tracker.lost_frames += 1
                if tracker.lost_frames > self.max_lost_frames:
                    tracker.active = False

        for i, detection in enumerate(detections):
            if i not in col_indices:
                self.trackers.append(Tracker(self.id_counter, detection[0], detection[1]))
                self.id_counter += 1

        self.trackers = [tracker for tracker in self.trackers if tracker.active]

    def draw_trackers(self, frame):
        for tracker in self.trackers:
            smoothed_positions = tracker.get_smoothed_positions()
            if len(smoothed_positions) >= 4:
                x = [p[0] for p in smoothed_positions]
                y = [p[1] for p in smoothed_positions]
                t = np.linspace(0, len(x) - 1, num=len(x))
                t_new = np.linspace(0, len(x) - 1, num=len(x) * 10)
                cs_x = CubicSpline(t, x)
                cs_y = CubicSpline(t, y)
                x_new = cs_x(t_new)
                y_new = cs_y(t_new)
                for i in range(1, len(x_new)):
                    cv2.line(frame, (int(x_new[i-1]), int(y_new[i-1])), (int(x_new[i]), int(y_new[i])), tracker.color, 2)
            else:
                for i in range(1, len(smoothed_positions)):
                    prev_pos = smoothed_positions[i - 1]
                    cur_pos = smoothed_positions[i]
                    cv2.line(frame, (int(prev_pos[0]), int(prev_pos[1])), (int(cur_pos[0]), int(cur_pos[1])), tracker.color, 2)
            print(f"Drawing trajectory for tracker {tracker.id} with {len(smoothed_positions)} positions.")


def find_tracker_by_position(trackers, position, velocity_threshold=VELOCITY_THRESHOLD, distance_threshold=DISTANCE_THRESHOLD):
    """Finds the closest tracker to the given position based on distance and velocity."""
    best_tracker = None
    min_cost = float('inf')  # Initialize with a large value
    
    for tracker in trackers:
        # Calculate the Euclidean distance between the tracker’s last known position and the detected position
        distance = np.linalg.norm(np.array(tracker.last_position) - np.array(position))
        
        # Optionally, include velocity in the cost calculation
        velocity_diff = np.linalg.norm(np.array([tracker.vel_x, tracker.vel_y]))

        if distance < distance_threshold:
            # Combine distance and velocity to determine the cost of associating the tracker with the detection
            cost = distance + velocity_diff  # You can weigh distance and velocity differently if needed
            
            # Update if this is the closest tracker so far
            if cost < min_cost:
                min_cost = cost
                best_tracker = tracker

    return best_tracker

class Tracker3D:
    """3D Tracker using Kalman filter to estimate position and velocity."""
    def __init__(self, id, initial_3d_position):
        self.id = id
        self.positions_3d = [initial_3d_position]

        # Initialize velocity
        self.vel_x = 0
        self.vel_y = 0
        self.vel_z = 0

        # Time step between frames
        dt = TIME_PER_FRAME

        # Kalman filter initialization
        self.kalman = cv2.KalmanFilter(6, 3)
        self.kalman.measurementMatrix = np.hstack((np.eye(3, dtype=np.float32), np.zeros((3, 3), dtype=np.float32)))
        self.kalman.transitionMatrix = np.array([
            [1, 0, 0, dt,  0,  0],
            [0, 1, 0,  0, dt,  0],
            [0, 0, 1,  0,  0, dt],
            [0, 0, 0,  1,  0,  0],
            [0, 0, 0,  0,  1,  0],
            [0, 0, 0,  0,  0,  1]
        ], dtype=np.float32)

        q = 1e-2  # Process noise factor
        self.kalman.processNoiseCov = q * np.array([
            [(dt**4)/4, 0, 0, (dt**3)/2, 0, 0],
            [0, (dt**4)/4, 0, 0, (dt**3)/2, 0],
            [0, 0, (dt**4)/4, 0, 0, (dt**3)/2],
            [(dt**3)/2, 0, 0, dt**2, 0, 0],
            [0, (dt**3)/2, 0, 0, dt**2, 0],
            [0, 0, (dt**3)/2, 0, 0, dt**2]
        ], dtype=np.float32)

        self.kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-1
        self.kalman.errorCovPost = np.eye(6, dtype=np.float32) * 1.0

        # Initialize state
        self.kalman.statePost = np.array([
            [initial_3d_position[0]],
            [initial_3d_position[1]],
            [initial_3d_position[2]],
            [0.1],  # Small initial velocity in X
            [0.1],  # Small initial velocity in Y
            [0.1]   # Small initial velocity in Z
        ], dtype=np.float32)

    def predict(self):
        """Predict the next state using the Kalman filter."""
        self.kalman.predict()

    def correct(self, measured_position_3d):
        """Update the state using the Kalman filter."""
        measurement = np.array([[measured_position_3d[0]],
                                [measured_position_3d[1]],
                                [measured_position_3d[2]]], dtype=np.float32)
        self.kalman.correct(measurement)

    def update_position(self, new_position_3d):
        """Update the position and velocity of the tracker."""
        self.predict()  # Predict the next state
        self.correct(new_position_3d)  # Correct with the measurement

        # Extract the updated state
        corrected_state = self.kalman.statePost.flatten()
        corrected_position = corrected_state[:3]
        corrected_velocity = corrected_state[3:]

        # Update tracker's position and velocity
        self.positions_3d.append(corrected_position.tolist())
        self.vel_x, self.vel_y, self.vel_z = corrected_velocity.tolist()


    def get_smoothed_positions_3D(self, window_size=5):
        """Smooth the positions using a simple moving average."""
        if len(self.positions_3d) < window_size:
            return self.positions_3d

        smoothed_positions_3D = []
        for i in range(len(self.positions_3d)):
            start_index = max(0, i - window_size + 1)
            end_index = i + 1
            window_positions = self.positions_3d[start_index:end_index]
            avg_x = sum(pos[0] for pos in window_positions) / len(window_positions)
            avg_y = sum(pos[1] for pos in window_positions) / len(window_positions)
            avg_z = sum(pos[2] for pos in window_positions) / len(window_positions)
            smoothed_positions_3D.append((avg_x, avg_y, avg_z))
        return smoothed_positions_3D

    def filter_outliers(self, max_velocity_threshold=MAX_VELOCITY_THRESHOLD):
        """Filter out unrealistic velocities."""
        if np.abs(self.vel_x) > max_velocity_threshold:
            self.vel_x *= 0.5  # Damp velocity instead of zeroing
        if np.abs(self.vel_y) > max_velocity_threshold:
            self.vel_y *= 0.5
        if np.abs(self.vel_z) > max_velocity_threshold:
            self.vel_z *= 0.5


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
    R1, R2, P1_rect, P2_rect, Q, _, _ = cv2.stereoRectify(K1, dist1, K2, dist2, image_size, R, T)

    # Initialize rectification maps for both cameras
    map1_left, map2_left = cv2.initUndistortRectifyMap(K1, dist1, R1, P1_rect, image_size, cv2.CV_32FC1)
    map1_right, map2_right = cv2.initUndistortRectifyMap(K2, dist2, R2, P2_rect, image_size, cv2.CV_32FC1)

    # Initialize video capture for both left and right cameras
    left_cap = cv2.VideoCapture(LEFT_VIDEO_PATH)
    right_cap = cv2.VideoCapture(RIGHT_VIDEO_PATH)

    # Initialize tracker managers for both left and right views
    manager_left = TrackerManager(OUTPUT_LEFT_CSV, OUTPUT_TRIANGULATION_CSV)
    manager_right = TrackerManager(OUTPUT_RIGHT_CSV, OUTPUT_TRIANGULATION_CSV)

    # Prepare video writer with high-quality settings
    width = int(left_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Original width
    height = int(left_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Original height
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    left_output = cv2.VideoWriter('/home/pratham/UNSW/camera_calib/stereo-camera-calibration/plots/flightgandhi/rec1/output_left1.avi', fourcc, FPS, (width, height))
    right_output = cv2.VideoWriter('/home/pratham/UNSW/camera_calib/stereo-camera-calibration/plots/flightgandhi/rec1/output_right1.avi', fourcc, FPS, (width, height))

    # Create directories to store output images
    output_left_images_dir = '/home/pratham/UNSW/camera_calib/stereo-camera-calibration/plots/flightgandhi/rec1/left_frames1/'
    output_right_images_dir = '/home/pratham/UNSW/camera_calib/stereo-camera-calibration/plots/flightgandhi/rec1/right_frames1/'
    os.makedirs(output_left_images_dir, exist_ok=True)
    os.makedirs(output_right_images_dir, exist_ok=True)

    # Dictionary to store 3D trackers by shared ID
    tracker_3d_dict = {}

    frame_no = 0
    total_left_blobs_detected = 0
    total_right_blobs_detected = 0
    total_blobs_matched = 0

    while True:
        ret_left, left_frame = left_cap.read()
        ret_right, right_frame = right_cap.read()

        if not ret_left or not ret_right:
            break

        # Apply rectification maps to both frames
        left_rectified = cv2.remap(left_frame, map1_left, map2_left, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_frame, map1_right, map2_right, cv2.INTER_LINEAR)

        # Detect blobs in both rectified frames
        left_blobs = detect_blobs(left_rectified)
        right_blobs = detect_blobs(right_rectified)

        total_left_blobs_detected += len(left_blobs)
        total_right_blobs_detected += len(right_blobs)

        for blob in left_blobs:
            cv2.circle(left_rectified, blob[0], BLOB_CIRCLE_RADIUS, (255, 0, 0), 2)
        for blob in right_blobs:
            cv2.circle(right_rectified, blob[0], BLOB_CIRCLE_RADIUS, (255, 0, 0), 2)

        # Compute matching scores and get matches
        matches = compute_matching_scores(left_blobs, right_blobs, F, manager_left.trackers, manager_right.trackers)
        total_blobs_matched += len(matches)

        # Update shared tracker IDs for left and right frames
        shared_tracker_ids = {}
        for match in matches:
            left_point, right_point = match['Left_Point'], match['Right_Point']
            shared_id = manager_left.id_counter
            shared_tracker_ids[shared_id] = (left_point, right_point)
            manager_left.id_counter += 1

        # Update trackers
        manager_left.update_trackers([(m['Left_Point'], m['Size_Difference']) for m in matches], frame_no)
        manager_right.update_trackers([(m['Right_Point'], m['Size_Difference']) for m in matches], frame_no)

        # Triangulate points and log to CSV
        if matches:
            # Triangulate 3D points
            points_3D = triangulate_points(matches, calib_data)
           
            # Update 3D trackers
            for shared_id, point_3D in zip(shared_tracker_ids.keys(), points_3D):
                if shared_id in tracker_3d_dict:
                    tracker_3d = tracker_3d_dict[shared_id]
                    tracker_3d.update_position(point_3D)
                else:
                    tracker_3d_dict[shared_id] = Tracker3D(shared_id, point_3D)
           
                tracker_3d = tracker_3d_dict[shared_id]
                print(f"Tracker {tracker_3d.id}: Position: {tracker_3d.positions_3d[-1]}, "
                      f"Velocity: ({tracker_3d.vel_x:.2f}, {tracker_3d.vel_y:.2f}, {tracker_3d.vel_z:.2f})")
           
            # Log triangulated points and velocities
            velocities = [[tracker.vel_x, tracker.vel_y, tracker.vel_z] for tracker in tracker_3d_dict.values()]
            sizes = [m['Size_Difference'] for m in matches]
            ids = list(shared_tracker_ids.keys())
            manager_left.log_triangulation_to_csv(points_3D, frame_no, sizes, velocities, ids)

        
        for blob in left_blobs:
            epiline = compute_epipolar_line(F, blob[0])
            draw_epipolar_line(right_rectified, epiline)
        # Draw matching lines
        draw_matching_lines(left_rectified, right_rectified, matches)
        manager_left.draw_trackers(left_rectified)
        manager_right.draw_trackers(right_rectified)

        # Save frames
        left_output.write(left_rectified)
        right_output.write(right_rectified)

        cv2.imshow('Left', left_rectified)
        cv2.imshow('Right', right_rectified)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_no += 1

    left_cap.release()
    right_cap.release()
    left_output.release()
    right_output.release()
    cv2.destroyAllWindows()

    print(f"Total Left Blobs Detected: {total_left_blobs_detected}")
    print(f"Total Right Blobs Detected: {total_right_blobs_detected}")
    print(f"Total Matched Blobs (Pairs): {total_blobs_matched}")


if __name__ == '__main__':
    main()

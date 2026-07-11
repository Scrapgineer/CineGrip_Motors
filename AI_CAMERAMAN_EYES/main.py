import cv2
import depthai as dai
import math
import blobconverter
import numpy as np

# ==========================================
# ZONE 1: CONFIGURATION & SMOOTHING TUNING
# ==========================================
DEADBAND_RADIUS = 25
TRACKING_TIMEOUT_FRAMES = 15

FACE_SMOOTHING = 0.2
TARGET_SMOOTHING = 0.1
MOTOR_SMOOTHING = 0.3

# ==========================================
# ZONE 2: OAK-D PIPELINE SETUP (v3 API)
# ==========================================
device = dai.Device(maxUsbSpeed=dai.UsbSpeed.HIGH)
pipeline = dai.Pipeline(device)

cam = pipeline.create(dai.node.Camera).build()
preview = cam.requestOutput((300, 300), type=dai.ImgFrame.Type.BGR888p)

face_nn = pipeline.create(dai.node.NeuralNetwork)
face_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=4))
preview.link(face_nn.input)

pose_nn = pipeline.create(dai.node.NeuralNetwork)
pose_nn.setBlobPath(blobconverter.from_zoo(name="head-pose-estimation-adas-0001", shaves=4))

# --- v3 QUEUE CREATION ---
# We no longer need XLinkIn/XLinkOut nodes. We create queues directly on the ports.
q_video = preview.createOutputQueue(maxSize=4, blocking=False)
q_face = face_nn.out.createOutputQueue(maxSize=4, blocking=False)
q_pose_out = pose_nn.out.createOutputQueue(maxSize=4, blocking=False)

# This single line replaces the entire XLinkIn setup
q_pose_in = pose_nn.input.createInputQueue(maxSize=4, blocking=False)

print("Starting P.A.C.O. Tracker v3.0... Press 'q' to quit.")
pipeline.start()

# ==========================================
# ZONE 3: STATE VARIABLES
# ==========================================
smooth_x, smooth_y = None, None
frames_lost = 0

smooth_tx, smooth_ty = 150.0, 150.0
desired_tx, desired_ty = 150.0, 150.0 

smooth_motor_pan, smooth_motor_tilt = 0.0, 0.0

def to_planar(arr: np.ndarray, shape: tuple):
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

# ==========================================
# ZONE 4: MAIN LOOP (v3 API)
# ==========================================
# In v3, since we called pipeline.start() above, we just check if it's running.
    # Keep running until 'q' is pressed
while True:
    # tryGet() instead of get() prevents the code from freezing if a frame is dropped
    video_msg = q_video.tryGet()
    if video_msg is None:
        continue
        
    img = video_msg.getCvFrame()
    h, w, _ = img.shape
    
    # Base Rule of Thirds
    w3, h3 = w // 3, h // 3
    
    # Draw Base Grid
    cv2.line(img, (w3, 0), (w3, h), (255, 255, 255), 1)
    cv2.line(img, (2*w3, 0), (2*w3, h), (255, 255, 255), 1)
    cv2.line(img, (0, h3), (w, h3), (255, 255, 255), 1)
    cv2.line(img, (0, 2*h3), (w, 2*h3), (255, 255, 255), 1)

    in_face = q_face.tryGet()
    primary_face = None
    max_area = 0

    if in_face is not None:
        detections = np.array(in_face.getFirstTensor()).reshape(-1, 7)
        for det in detections:
            conf = det[2]
            if conf > 0.5:
                x1 = max(0, int(det[3] * w))
                y1 = max(0, int(det[4] * h))
                x2 = min(w, int(det[5] * w))
                y2 = min(h, int(det[6] * h))
                
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    primary_face = (x1, y1, x2, y2)

    if primary_face:
        frames_lost = 0
        x1, y1, x2, y2 = primary_face
        
        # --- Aspect Ratio Preservation Fix ---
        fw, fh = x2 - x1, y2 - y1
        sq_size = max(fw, fh)
        cx, cy = x1 + fw // 2, y1 + fh // 2
        
        sq_x1 = max(0, cx - sq_size // 2)
        sq_y1 = max(0, cy - sq_size // 2)
        sq_x2 = min(w, cx + sq_size // 2)
        sq_y2 = min(h, cy + sq_size // 2)
        
        face_crop = img[sq_y1:sq_y2, sq_x1:sq_x2]
        
        if face_crop.size > 0:
            # In DepthAI v3, we send host images using ImgFrame instead of NNData
            img_frame = dai.ImgFrame()
            img_frame.setType(dai.ImgFrame.Type.BGR888p) # Planar BGR format expected by the NN
            img_frame.setWidth(60)
            img_frame.setHeight(60)
            img_frame.setData(to_planar(face_crop, (60, 60)))
            q_pose_in.send(img_frame)
            
            in_pose = q_pose_out.get()
            
            # In DepthAI v3, "Layers" were renamed to "Tensors"
            yaw_tensor = np.array(in_pose.getTensor('angle_y_fc')).flatten()
            pitch_tensor = np.array(in_pose.getTensor('angle_p_fc')).flatten()

            # Safely extract the angle (handling raw bytes if the API hasn't decoded them)
            yaw = yaw_tensor.view(np.float16)[0] if yaw_tensor.dtype == np.uint8 else yaw_tensor[0]
            pitch = pitch_tensor.view(np.float16)[0] if pitch_tensor.dtype == np.uint8 else pitch_tensor[0]
            
            # --- Center-Return Logic (Default State) ---
            desired_tx = w // 2   # Default Horizontal Center
            desired_ty = h3       # Default Top Third
            target_name = "Top (Centered)"
            
            # --- Horizontal Lead Room (FIXED: True Cinematic Gaze Space) ---
            if yaw > 15:  
                # This triggers when you look to your actual RIGHT
                desired_tx = w3      # Places you on the Left Third, giving you lead room on the right
                target_name = "Left Third (Looking Right)"
            elif yaw < -15: 
                # This triggers when you look to your actual LEFT
                desired_tx = 2 * w3  # Places you on the Right Third, giving you lead room on the left
                target_name = "Right Third (Looking Left)"
                
            # --- Vertical Lead Room (Pitch Logic) ---
            if pitch < -15: # Looking Up
                desired_ty = 2 * h3  # Target bottom third for top lead room
                target_name += " & Bottom"
            elif pitch > 15: # Looking Down
                desired_ty = h // 4  # Push slightly higher than Top Third
                target_name += " & High"
                
            # --- Dynamic Framing by Shot Size ---
            # If face is larger than 35% of frame height (Close Up)
            if fh > (h * 0.35):
                # Pull the Y target 30% closer to the vertical center to save the chin
                desired_ty = int(desired_ty * 0.7 + (h // 2) * 0.3)
                target_name += " (Close-up)"

        # Target the eyes (upper third of face)
        raw_target_x = (x1 + x2) // 2
        raw_target_y = y1 + ((y2 - y1) // 3) 
        
        if smooth_x is None:
            smooth_x, smooth_y = raw_target_x, raw_target_y
        else:
            smooth_x = int((raw_target_x * FACE_SMOOTHING) + (smooth_x * (1 - FACE_SMOOTHING)))
            smooth_y = int((raw_target_y * FACE_SMOOTHING) + (smooth_y * (1 - FACE_SMOOTHING)))
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(img, (smooth_x, smooth_y), 5, (0, 0, 255), -1)

    else:
        frames_lost += 1
        target_name = "COASTING"
        if frames_lost > TRACKING_TIMEOUT_FRAMES:
            smooth_x, smooth_y = None, None
            smooth_motor_pan, smooth_motor_tilt = 0, 0
            target_name = "SEARCHING..."

    if smooth_x is not None:
        # Framing Target Smoothing
        smooth_tx = (smooth_tx * (1 - TARGET_SMOOTHING)) + (desired_tx * TARGET_SMOOTHING)
        smooth_ty = (smooth_ty * (1 - TARGET_SMOOTHING)) + (desired_ty * TARGET_SMOOTHING)
        
        cv2.drawMarker(img, (int(smooth_tx), int(smooth_ty)), (255, 0, 0), cv2.MARKER_CROSS, 15, 2)
        
        error_x = smooth_x - smooth_tx
        error_y = smooth_y - smooth_ty
        min_dist = math.sqrt(error_x**2 + error_y**2)
        
        cv2.circle(img, (int(smooth_tx), int(smooth_ty)), DEADBAND_RADIUS, (100, 100, 100), 1)

        if min_dist <= DEADBAND_RADIUS:
            raw_motor_pan, raw_motor_tilt = 0, 0
            line_color = (0, 255, 0)
            status_text = "LOCKED"
        else:
            raw_motor_pan = error_x
            raw_motor_tilt = error_y
            line_color = (0, 255, 255)
            status_text = "TRACKING"
        
        smooth_motor_pan = (raw_motor_pan * MOTOR_SMOOTHING) + (smooth_motor_pan * (1 - MOTOR_SMOOTHING))
        smooth_motor_tilt = (raw_motor_tilt * MOTOR_SMOOTHING) + (smooth_motor_tilt * (1 - MOTOR_SMOOTHING))
        
        cv2.line(img, (smooth_x, smooth_y), (int(smooth_tx), int(smooth_ty)), line_color, 2)
        
        cv2.putText(img, f"Target: {target_name} [{status_text}]", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2)
        cv2.putText(img, f"Pan Cmd: {int(smooth_motor_pan)} | Tilt Cmd: {int(smooth_motor_tilt)}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2)
    else:
        cv2.putText(img, "Looking for hoes...", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("P.A.C.O. VISION", img)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
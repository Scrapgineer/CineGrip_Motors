import cv2
import depthai as dai
import math
import blobconverter
import numpy as np

# Initialize device and pipeline
device = dai.Device(maxUsbSpeed=dai.UsbSpeed.HIGH)
pipeline = dai.Pipeline(device)

# Initialize camera node
cam = pipeline.create(dai.node.Camera).build()
preview = cam.requestOutput((300, 300), type=dai.ImgFrame.Type.BGR888p)

# Initialize Neural Network node
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=5))
preview.link(nn.input)

# Create output queues
q_video = preview.createOutputQueue(maxSize=4, blocking=False)
q_nn = nn.out.createOutputQueue(maxSize=4, blocking=False)

print("Starting Cinematic Tracker... Press 'q' to quit.")
pipeline.start()

# Main loop
with pipeline:
    while pipeline.isRunning():
        img = q_video.get().getCvFrame()
        h, w, _ = img.shape
        
        # Calculate rule of thirds intersections
        w3, h3 = w // 3, h // 3
        intersections = {
            "Top Left": (w3, h3),
            "Top Right": (2*w3, h3),
            "Bottom Left": (w3, 2*h3),
            "Bottom Right": (2*w3, 2*h3)
        }
        
        # Retrieve NN data
        in_nn = q_nn.get()
        raw_data = in_nn.getFirstTensor()
        detections = np.array(raw_data).reshape(-1, 7)
        
        # Draw grid
        cv2.line(img, (w3, 0), (w3, h), (255, 255, 255), 1)
        cv2.line(img, (2*w3, 0), (2*w3, h), (255, 255, 255), 1)
        cv2.line(img, (0, h3), (w, h3), (255, 255, 255), 1)
        cv2.line(img, (0, 2*h3), (w, 2*h3), (255, 255, 255), 1)

        # Draw deadband zones
        DEADBAND_RADIUS = 25
        for name, (ix, iy) in intersections.items():
            cv2.circle(img, (ix, iy), DEADBAND_RADIUS, (100, 100, 100), 1)

        # Process face detections
        for det in detections:
            conf = det[2]
            
            if conf > 0.4:
                x1, y1 = int(det[3] * w), int(det[4] * h)
                x2, y2 = int(det[5] * w), int(det[6] * h)
                
                target_x = (x1 + x2) // 2
                target_y = y1 + ((y2 - y1) // 3) 
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(img, (target_x, target_y), 5, (0, 0, 255), -1)

                min_dist = float('inf')
                target_name = ""

                for name, (ix, iy) in intersections.items():
                    d = math.sqrt((target_x - ix)**2 + (target_y - iy)**2)
                    if d < min_dist:
                        min_dist = d
                        target_name = name
                
                tx, ty = intersections[target_name]
                
                # Evaluate deadband logic
                if min_dist <= DEADBAND_RADIUS:
                    motor_error = 0
                    line_color = (0, 255, 0)
                    status_text = "LOCKED"
                else:
                    motor_error = int(min_dist - DEADBAND_RADIUS)
                    line_color = (0, 255, 255)
                    status_text = "TRACKING"
                
                # UI Overlay
                cv2.putText(img, f"Target: {target_name} [{status_text}]", (10, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2)
                cv2.putText(img, f"Motor Error: {motor_error}px", (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2)
                
                # Draw error line
                cv2.line(img, (target_x, target_y), (tx, ty), line_color, 2)

        cv2.imshow("AI CAM", img)
        if cv2.waitKey(1) == ord('q'):
            break

# Clean up windows
cv2.destroyAllWindows()
import cv2
import depthai as dai
import math
import blobconverter
import numpy as np

# Setup
device = dai.Device(maxUsbSpeed=dai.UsbSpeed.HIGH)
pipeline = dai.Pipeline(device)

cam = pipeline.create(dai.node.Camera).build()
preview = cam.requestOutput((300, 300), type=dai.ImgFrame.Type.BGR888p)

# AI Config (Raw NN Node)
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=5))
preview.link(nn.input)

# Queues
q_video = preview.createOutputQueue(maxSize=4, blocking=False)
q_nn = nn.out.createOutputQueue(maxSize=4, blocking=False)

print("Starting Cinematic Tracker... Press 'q' to quit.")
pipeline.start()

# Rule of Thirds points
intersections = {
    "Top Left": (100, 100),
    "Top Right": (200, 100),
    "Bottom Left": (100, 200),
    "Bottom Right": (200, 200)
}

# Main loop
with pipeline:
    while pipeline.isRunning():
        # Get frame
        img = q_video.get().getCvFrame()
        h, w, _ = img.shape
        
        # Get raw NN data
        in_nn = q_nn.get()
        
        # LA SOLUCIÓN DIRECTA: Usar getFirstTensor()
        raw_data = in_nn.getFirstTensor()
        detections = np.array(raw_data).reshape(-1, 7)
        
        # Draw grid
        cv2.line(img, (w//3, 0), (w//3, h), (255, 255, 255), 1)
        cv2.line(img, (2*w//3, 0), (2*w//3, h), (255, 255, 255), 1)
        cv2.line(img, (0, h//3), (w, h//3), (255, 255, 255), 1)
        cv2.line(img, (0, 2*h//3), (w, 2*h//3), (255, 255, 255), 1)

        # Process detections
        for det in detections:
            conf = det[2]
            
            # Confidence filter (40%)
            if conf > 0.4:
                # Scale coords
                x1, y1 = int(det[3] * w), int(det[4] * h)
                x2, y2 = int(det[5] * w), int(det[6] * h)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Draw target
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

                min_dist = float('inf')
                target_name = ""

                for name, (ix, iy) in intersections.items():
                    # Pythagoras distance
                    d = math.sqrt((cx - ix)**2 + (cy - iy)**2)
                    if d < min_dist:
                        min_dist = d
                        target_name = name
                
                # UI Overlay
                cv2.putText(img, f"Target: {target_name}", (10, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(img, f"Error: {int(min_dist)}px", (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Error line
                tx, ty = intersections[target_name]
                cv2.line(img, (cx, cy), (tx, ty), (0, 255, 255), 2)

        cv2.imshow("AI CAM", img)
        if cv2.waitKey(1) == ord('q'):
            break
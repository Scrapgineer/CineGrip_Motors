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

print("Starting Cinematic Tracker (Step 1)... Press 'q' to quit.")
pipeline.start()

# Main loop
with pipeline:
    while pipeline.isRunning():
        # Get frame
        img = q_video.get().getCvFrame()
        h, w, _ = img.shape
        
        # --- PASO 1A: INTERSECCIONES DINÁMICAS ---
        # Calculamos los tercios exactos en base al tamaño real del video
        w3, h3 = w // 3, h // 3
        intersections = {
            "Top Left": (w3, h3),
            "Top Right": (2*w3, h3),
            "Bottom Left": (w3, 2*h3),
            "Bottom Right": (2*w3, 2*h3)
        }
        
        # Get raw NN data
        in_nn = q_nn.get()
        raw_data = in_nn.getFirstTensor()
        detections = np.array(raw_data).reshape(-1, 7)
        
        # Draw grid
        cv2.line(img, (w3, 0), (w3, h), (255, 255, 255), 1)
        cv2.line(img, (2*w3, 0), (2*w3, h), (255, 255, 255), 1)
        cv2.line(img, (0, h3), (w, h3), (255, 255, 255), 1)
        cv2.line(img, (0, 2*h3), (w, 2*h3), (255, 255, 255), 1)

        # Process detections
        for det in detections:
            conf = det[2]
            
            # Confidence filter (40%)
            if conf > 0.4:
                # Scale coords
                x1, y1 = int(det[3] * w), int(det[4] * h)
                x2, y2 = int(det[5] * w), int(det[6] * h)
                
                # --- PASO 1B: HEADROOM (ALTURA DE LOS OJOS) ---
                # En lugar de sacar el centro Y (y1 + y2 // 2), calculamos un tercio hacia abajo 
                # desde la parte superior de la caja de detección.
                target_x = (x1 + x2) // 2
                target_y = y1 + ((y2 - y1) // 3) 
                
                # Draw target
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(img, (target_x, target_y), 5, (0, 0, 255), -1) # Punto rojo en los ojos

                min_dist = float('inf')
                target_name = ""

                for name, (ix, iy) in intersections.items():
                    # Pythagoras distance (usando los nuevos target_x y target_y)
                    d = math.sqrt((target_x - ix)**2 + (target_y - iy)**2)
                    if d < min_dist:
                        min_dist = d
                        target_name = name
                
                # UI Overlay
                cv2.putText(img, f"Target: {target_name}", (10, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(img, f"Error: {int(min_dist)}px", (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Error line (ahora sale desde los ojos)
                tx, ty = intersections[target_name]
                cv2.line(img, (target_x, target_y), (tx, ty), (0, 255, 255), 2)

        cv2.imshow("AI CAM", img)
        if cv2.waitKey(1) == ord('q'):
            break
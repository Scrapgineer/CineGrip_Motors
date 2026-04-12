import cv2
import depthai as dai

device = dai.Device(maxUsbSpeed=dai.UsbSpeed.HIGH)

pipeline = dai.Pipeline(device)

cam = pipeline.create(dai.node.Camera).build()

camera_output = cam.requestOutput((640, 360), type=dai.ImgFrame.Type.BGR888p)

video_queue = camera_output.createOutputQueue()

print("Starting AI Cameraman... Press 'q' on the video window to quit.")

pipeline.start()

with pipeline:
    while pipeline.isRunning():
        videoIn = video_queue.get()
        frame = videoIn.getCvFrame()

        height, width, _ = frame.shape

        third_x = width // 3
        third_y = height // 3

        cv2.line(frame, (third_x, 0), (third_x, height), (255, 255, 255), 2)
        cv2.line(frame, (third_x * 2, 0), (third_x * 2, height), (255, 255, 255), 2)

        cv2.line(frame, (0, third_y), (width, third_y), (255, 255, 255), 2)
        cv2.line(frame, (0, third_y * 2), (width, third_y * 2), (255, 255, 255), 2)

        cv2.imshow("OAK-D Lite - AI Cameraman", frame)

        if cv2.waitKey(1) == ord('q'):
            break
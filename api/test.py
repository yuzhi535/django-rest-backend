import os
import numpy as np
import cv2 as cv
import av


fps = 5

container = av.open("test.mp4", mode="w")

stream = container.add_stream("h264", rate=fps)
stream.pix_fmt = "yuv420p"

for file in os.listdir('../course/taiji/pro'):

    if file[-3:] == 'jpg':
        img = cv.imread(os.path.join('../course/taiji/pro', file))
        stream.height, stream.width = img.shape[:-1]
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

# Flush stream
for packet in stream.encode():
    container.mux(packet)

# Close the file
container.close()
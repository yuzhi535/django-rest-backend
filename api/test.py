from vidgear.gears import CamGear
from imutils.video import FileVideoStream
from imutils.video import FPS
import cv2

# open any valid video stream(for e.g `myvideo.avi` file)
stream = CamGear(source="../media/123/taiji/test.mp4").start()
# stream = FileVideoStream(path='../media/123/taiji/test.mp4').start()
fps = FPS().start()
# loop over
while True:

    # read frames from stream
    frame = stream.read()
    fps.update()
    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

    # Show output window
    cv2.imshow("Output", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
fps.stop()
print(f'fps: {fps.fps(), fps.elapsed()}')
# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()
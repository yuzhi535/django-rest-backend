from typing import List, Mapping, Optional, Tuple, Union
import av
import cv2 as cv
import os
import time
import mediapipe as mp
from imutils.video import FileVideoStream
from mediapipe.python.solutions.drawing_utils import RED_COLOR, DrawingSpec, _RGB_CHANNELS, WHITE_COLOR, \
    _normalized_to_pixel_coordinates
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
position = list(range(11, 33)) + [0, ]
draw_position = position + [2, 5, 7, 8]


POSE_CONNECTIONS = frozenset([(0, 2), (0, 5), (2, 7), (5, 8), (11, 12), (11, 13),
                              (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                              (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                              (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32)])
_VISIBILITY_THRESHOLD = 0.5
_PRESENCE_THRESHOLD = 0.5


def predict(image: Union[str, np.ndarray]):

    BG_COLOR = (192, 192, 192)

    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
        coordinates = {}
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            return None

        annotated_image = image.copy()
        # Draw segmentation on the image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        # condition = np.stack(
        #     (results.segmentation_mask,) * 3, axis=-1) > 0.1
        # bg_image = np.zeros(image.shape, dtype=np.uint8)
        # bg_image[:] = BG_COLOR
        # annotated_image = np.where(
        #     condition, annotated_image, bg_image)
        # Draw pose landmarks on the image.
        draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    results = {
        'coordinates': coordinates,
        'data': annotated_image
    }
    return results


def draw_landmarks(image, landmark_list, connections: Optional[List[Tuple[int, int]]] = None, landmark_drawing_spec: Union[DrawingSpec,
                                                                                                                           Mapping[int, DrawingSpec]] = DrawingSpec(
    color=RED_COLOR),
    connection_drawing_spec: Union[DrawingSpec,
                                   Mapping[Tuple[int, int],
                                           DrawingSpec]] = DrawingSpec()):

    if not landmark_list:
        return
    if image.shape[2] != _RGB_CHANNELS:
        raise ValueError(
            'Input image must contain three channel rgb data.')
    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if idx not in draw_position:
            continue
        if ((landmark.HasField('visibility') and
                landmark.visibility < _VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and
                    landmark.presence < _PRESENCE_THRESHOLD)):
            continue
        landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                       image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    if connections:
        num_landmarks = len(landmark_list.landmark)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection '
                                 f'from landmark #{start_idx} to landmark #{end_idx}.')
            if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                drawing_spec = connection_drawing_spec[connection] if isinstance(
                    connection_drawing_spec, Mapping) else connection_drawing_spec
                cv.line(image, idx_to_coordinates[start_idx],
                        idx_to_coordinates[end_idx], drawing_spec.color,
                        drawing_spec.thickness)
    # Draws landmark points after finishing the connection lines, which is
    # aesthetically better.
    if landmark_drawing_spec:
        for idx, landmark_px in idx_to_coordinates.items():
            drawing_spec = landmark_drawing_spec[idx] if isinstance(
                landmark_drawing_spec, Mapping) else landmark_drawing_spec
            # White circle border
            circle_border_radius = max(drawing_spec.circle_radius + 1,
                                       int(drawing_spec.circle_radius * 1.2))
            cv.circle(image, landmark_px, circle_border_radius, WHITE_COLOR,
                      drawing_spec.thickness)
            # Fill color into the circle
            cv.circle(image, landmark_px, drawing_spec.circle_radius,
                      drawing_spec.color, drawing_spec.thickness)


def make_video(imgs, output_path, filename):
    assert len(imgs) > 0, 'the images count is zero!!!'

    img = imgs[0]
    fps = 5

    container = av.open(os.path.join(output_path, filename), mode="w")
    print(f'file path is {os.path.join(output_path, filename)}')
    stream = container.add_stream("h264", rate=fps)
    stream.pix_fmt = "yuv420p"
    stream.height, stream.width = img.shape[:-1]

    for img in imgs:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush stream
    for packet in stream.encode():
        container.mux(packet)

    # Close the file
    container.close()


def split_video(video: str, fps_limit=10) -> list:
    """
    split a video to frames. return a list of a quantity of images
    use imutils video library, a multi-thread opencv method
    """

    stream = FileVideoStream(video).start()

    imgs = []

    elapsed = 1 / fps_limit
    prev = time.time()
    i = 0
    print('start process video')
    while stream.more():
        # grab the frame from the threaded video file stream
        frame = stream.read()

        elapsed_time = time.time() - prev
        if elapsed_time > elapsed:
            prev = time.time()
            imgs.append(frame)
            i += 1
    stream.stop()
    print('end process video')
    print(f'num of frames i {len(imgs)}')
    return imgs

def create():
    base_path = 'media/taiji/0_qishi.mp4'
    imgs = split_video(base_path, fps_limit=100)
    out_imgs = []
    for img in imgs:
        out_img = predict(img)['data']
        out_imgs.append(out_img)
    
    make_video(out_imgs, output_path='media/taiji', filename='test.mp4')

if __name__ == '__main__':
    base_dir='course/taiji/pro'
    path='{i}_res.jpg'
    imgs=[]
    for idx in range(len(os.listdir(base_dir))//3):
        file_path=os.path.join(base_dir, path.format(i=idx))
        img=cv.imread(file_path)
        imgs.append(img)
    
    make_video(imgs, output_path='media/taiji', filename='test.mp4')
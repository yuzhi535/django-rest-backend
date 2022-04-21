from __future__ import division

import math
import os
import time
from typing import Union, Optional, Tuple, Mapping, List

import cv2 as cv
import mediapipe as mp
import numpy as np
import pandas as pd
from django.contrib.auth import authenticate
from django.contrib.auth.hashers import make_password
from django.db import IntegrityError
from django.http import HttpResponse, JsonResponse
# deal with videos
from imutils.video import FileVideoStream
from mediapipe.python.solutions.drawing_utils import RED_COLOR, DrawingSpec, _RGB_CHANNELS, WHITE_COLOR, \
    _normalized_to_pixel_coordinates
from rest_framework import views
from rest_framework.authtoken.models import Token
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.decorators import api_view
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.views import APIView

from backend.models import User, CustomUser

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

division = 10000
_VISIBILITY_THRESHOLD = 0.5
_PRESENCE_THRESHOLD = 0.5
POSE_CONNECTIONS = frozenset([(0, 2), (0, 5), (2, 7), (5, 8), (11, 12), (11, 13),
                              (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                              (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                              (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32)])


def show(request):
    return HttpResponse("fucku")


@api_view(['POST'])
def login(request):
    password = request.POST.get('password')
    phone_number = request.POST.get('phone_number')
    user = authenticate(phone_number=phone_number, password=password)
    if not user:
        try:
            if not CustomUser.objects.get(phone_number=phone_number):
                return JsonResponse({'status': 'A404'})  # 用户不存在
            else:
                return JsonResponse({'status': 'B404'})  # 密码输入错误
        except CustomUser.DoesNotExist:
            return JsonResponse({'status': 'A404'})
    else:
        name = User.objects.get(user_id=user.id).name
        return JsonResponse({'id': user.id,
                             'name': name,
                             'status': 200})


@api_view(['POST'])
def getavatar(request):
    phone_number = request.POST.get('phone_number')
    user_id = CustomUser.objects.get(phone_number=phone_number).id
    avatar = User.objects.get(user_id=user_id).avatar
    return JsonResponse(data={'avatar': f'{avatar}'})


@api_view(['POST'])
def register(request):
    phone_number = request.POST.get('phone_number')
    try:
        password1 = request.POST.get('password1')
        password1 = password_long(password1)
    except Exception as e:
        return HttpResponse(content=f'{format(e)}')
    password2 = request.POST.get('password2')
    name = request.POST.get('name')
    gender = request.POST.get('gender')
    try:
        height = request.POST.get('height')
        height = more_than(height)
    except Exception as e:
        return HttpResponse(content=f'{format(e)}')
    try:
        weight = request.POST.get('weight')
        weight = more_than(weight)
    except Exception as e:
        return HttpResponse(content=f'{format(e)}')
    avatar = request.FILES.get('avatar')
    birthday = request.POST.get('birthday')
    idcard_number = request.POST.get('idcard_number')
    hobbies = request.POST.get('hobbies')

    user = CustomUser.objects.filter(phone_number=phone_number)
    if user:
        return JsonResponse({'status': 'C400'})
    else:
        if password1 == password2:
            password = make_password(password1)
            try:
                instance1 = CustomUser.objects.create(
                    phone_number=phone_number,
                    password=password,
                    gender=gender,
                )
                instance2 = User.objects.create(
                    user=instance1,
                    name=name,
                    height=height,
                    weight=weight,
                    birthday=birthday,
                    hobbies=hobbies,
                    idcard_number=idcard_number,
                    avatar=avatar,
                )
            except(TypeError, ValueError, IntegrityError):
                CustomUser.objects.get(phone_number=phone_number).delete()
                return JsonResponse({'status': 'A400'})
            else:
                return JsonResponse(data={
                    "phone_number": instance1.phone_number,
                    "password": instance1.password,
                    "gender": instance1.gender,
                    "id": instance2.user_id,
                    "name": instance2.name,
                    "height": instance2.height,
                    "weight": instance2.weight,
                    "birthday": instance2.birthday,
                    "hobbies": instance2.hobbies,
                    "idcard_number": instance2.idcard_number,
                    "avatar_status": 'taiji' if instance2.avatar else '0',
                    "status": '200'
                })
        else:
            return JsonResponse({'status': 'B400'})


# 密码小于6位异常
def password_long(pwd):
    if len(pwd) >= 6:
        return pwd
    else:
        pwd_error = Exception('密码长度不能小于6位')
        raise pwd_error


# 身高体重必须大于0异常
def more_than(hw):
    if hw:
        if int(hw) > 0:
            return hw
        else:
            zero_error = Exception('身高或体重不可以是负数')
            raise zero_error


class FileUploadView(views.APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, filename, format=None):
        # file_obj = request.data['file']
        file_obj = request.FILES.get("file")
        course = request.POST.get('course')
        userID = request.POST.get('userID')
        try:
            assert course is not None or not ""  # have to do this!!!
            assert userID is not None or not ""
            assert file_obj is not None
        except AssertionError:
            return JsonResponse({'status': 401})

        path = os.path.join(os.getcwd(), 'media')
        if not os.path.exists(path):
            os.mkdir(path)
        if not os.path.exists(os.path.join(path, userID)):
            os.mkdir(os.path.join(path, userID))
        if not os.path.exists(os.path.join(path, userID, course)):
            os.mkdir(os.path.join(path, userID, course))
        outpath = os.path.join(os.path.join(path, userID, course, filename))
        # if os.listdir():
        #     count = len([lists for lists in os.listdir(os.path.join(userID, course)) if
        #                  os.path.isfile(os.path.join(os.path.join(userID, course), lists))])
        #     outpath = str(count) + outpath

        # save the video
        with open(outpath, 'wb') as f:
            for chunk in file_obj.chunks():
                f.write(chunk)
        return JsonResponse({'status': 204})


class CustomAuthToken(ObtainAuthToken):
    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data,
                                           context={'request': request})
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data['user']
        token, created = Token.objects.get_or_create(user=user)
        return Response({
            'token': token.key,
            'user_id': user.pk,
        })


'''
1. 首先要兼容openpose生成的csv文件
2. 
'''


class Predict(APIView):
    position = list(range(11, 33)) + [0, ]
    draw_position = position + [2, 5, 7, 8]

    position_map = {
        11: 'Left shoulder',
        12: 'Right shoulder',
        13: 'Left elbow',
        14: 'Right elbow',
        15: 'Left wrist',
        16: 'Right wrist',
        17: 'Left pinky ',  # 1 knuckle
        18: 'Right pinky',  # 1 knuckle
        19: 'Left index',  # 1 knuckle
        20: 'Right index',  # 1 knuckle
        21: 'Left thumb',  # 2 knuckle
        22: 'Right thumb',  # 2 knuckle
        23: 'Left hip',
        24: 'Right hip',
        25: 'Left knee',
        26: 'Right knee',
        27: 'Left ankle',
        28: 'Right ankle',
        29: 'Left heel',
        30: 'Right heel',
        31: 'Left foot index',
        32: 'Right foot index',
        0: 'nose',
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def transform(data, center):
        for idx, (name, point) in enumerate(data.items()):
            if point.any():
                point -= center

    def post(self, request):
        course = request.POST.get('course')
        userID = request.POST.get('userID')
        filename = request.POST.get('filename')

        uc_path = os.path.join('media', userID, course)
        split_path = os.path.join(uc_path, 'test_split')
        uv_path = os.path.join(uc_path, filename)
        pro_path = os.path.join(uc_path, 'pro')
        imgs = self.split_video(uv_path, output_path=split_path, fps_limit=100)

        res = self.process(imgs, output_path=pro_path)

        try:
            self.make_video(res['data'], uc_path, filename)
        except AttributeError:
            return JsonResponse({'status': 403})

        out = Predict.post_process(res=res)
        criterion = self.process_csv(
            base_path=os.path.join('course', course, 'pro'))
        cost_rarm = self.dynamic_time_warping(criterion[0], out[0])
        cost_larm = self.dynamic_time_warping(criterion[1], out[1])
        cost_rleg = self.dynamic_time_warping(criterion[2], out[2])
        cost_lleg = self.dynamic_time_warping(criterion[3], out[3])
        cost_head = self.dynamic_time_warping(criterion[4], out[4])

        print('head', cost_head)
        print('rarm', cost_rarm)
        print('larm', cost_larm)
        print('rleg', cost_rleg)
        print('lleg', cost_lleg)
        # score = 100 - max(cost_head, cost_larm,
        #                   cost_larm, cost_rleg, cost_lleg)

        key = 1.5
        evaluate = ""
        error = ["您的头部动作不标准 ", "您的右臂动作不标准 ",
                 "您的左臂动作不标准 ", "您的右腿动作不标准 ", "您的左腿动作不标准 "]
        if cost_head >= key:
            evaluate += error[0]
        if cost_rarm >= key:
            evaluate = evaluate + error[1]
        if cost_larm >= key:
            evaluate = evaluate + error[2]
        if cost_rleg >= key:
            evaluate = evaluate + error[3]
        if cost_lleg >= key:
            evaluate = evaluate + error[4]

        if evaluate == "":
            evaluate = "您的动作非常完美"

        filename = filename[:-4] + "_e.mp4"
        return JsonResponse(data={'status': 204,
                                  'evaluate': evaluate,
                                  'url': f'{userID}/{course}/{filename}'})

    def delete_all_files(self, dir: str):
        for file in os.listdir(dir):
            os.remove(os.path.join(dir, file))

    @staticmethod
    def post_process(res, openpose=False, thresh=0.3):
        rarms = []
        larms = []
        rlegs = []
        llegs = []
        head = []

        if not openpose:
            for coordinates in res['coordinates']:
                pos = {}

                # 只需要看到14个就够了，不需要脸部数据
                for idx in Predict.position:
                    if coordinates.get(Predict.position_map[idx]) is None:
                        pos[Predict.position_map[idx]] = np.array([np.NAN, np.NAN])
                        continue
                    elif coordinates[Predict.position_map[idx]][2] < thresh:
                        pos[Predict.position_map[idx]] = np.array([np.NAN, np.NAN])
                        continue
                    else:
                        pos[Predict.position_map[idx]] = np.array(
                            [int(coordinates[Predict.position_map[idx]][0]),
                             int(coordinates[Predict.position_map[idx]][1])])

                # 以左肩膀和右肩膀的中点为中心
                if not np.any(np.array([np.isnan(np.min(pos[Predict.position_map[11]])),
                                        np.isnan(np.min(pos[Predict.position_map[12]]))])):
                    center = (pos[Predict.position_map[11]] + pos[Predict.position_map[12]]) // 2
                    Predict.transform(pos, center)
                else:
                    continue

                rarms.append({'Right shoulder': pos.get(Predict.position_map[12]), 'Right elbow': pos.get(
                    Predict.position_map[14]), 'Right thumb': pos.get(Predict.position_map[22])})
                larms.append({'Left shoulder': pos.get(Predict.position_map[11]), 'Left elbow': pos.get(
                    Predict.position_map[13]), 'Left thumb': pos.get(Predict.position_map[21])})
                rlegs.append({'Right hip': pos.get(Predict.position_map[24]), 'Right knee': pos.get(
                    Predict.position_map[26]), 'Right ankle': pos.get(Predict.position_map[28])})
                llegs.append({'Left hip': pos.get(Predict.position_map[13]), 'Left knee': pos.get(
                    Predict.position_map[15]), 'Left ankle': pos.get(Predict.position_map[27])})
                head.append(
                    {'nose': pos.get(Predict.position_map[0])}
                )
        else:
            candidates, subsets = res[0], res[1]
            pos_name = ('nose', 'neck',
                        'rshoulder', 'relbow', 'rhand',
                        'lshoulder', 'lelbow', 'lhand',
                        'rhip', 'rknee', 'rankle',
                        'lhip', 'lknee', 'lankle')

            for i in range(len(candidates)):
                candidate_dat = candidates[i]
                subset_dat = subsets[i]
                if candidate_dat.empty:
                    print('无人')
                    continue
                if subset_dat.shape[0] > 1:
                    print('多人！')
                    continue

                pos = {}

                # 只需要看到14个就够了，不需要脸部数据
                for idx in range(1, 15):
                    k = subset_dat.loc[0][idx]
                    if k == -1:
                        pos[pos_name[idx - 1]] = np.array([np.NAN, np.NAN])
                        continue
                    elif candidate_dat.loc[k][3] < thresh:
                        pos[pos_name[idx - 1]] = np.array([np.NAN, np.NAN])
                        continue
                    else:
                        pos[pos_name[idx - 1]] = np.array(
                            [int(candidate_dat.loc[k][0]), int(candidate_dat.loc[k][1])])
                if not np.isnan(np.min(pos['neck'])):
                    Predict.transform(pos, pos['neck'].copy())

                rarms.append({'Right shoulder': pos.get('rshoulder'), 'Right elbow': pos.get(
                    'relbow'), 'Right thumb': pos.get('rhand')})
                larms.append({'Left shoulder': pos.get('lshoulder'), 'Left elbow': pos.get(
                    'lelbow'), 'Left thumb': pos.get('lhand')})
                rlegs.append({'Right hip': pos.get('rhip'), 'Right knee': pos.get(
                    'rknee'), 'Right ankle': pos.get('rankle')})
                llegs.append({'Left hip': pos.get('lhip'), 'Left knee': pos.get(
                    'lknee'), 'Left ankle': pos.get('lankle')})
                head.append(
                    {'nose': pos.get('nose')}
                )
        return rarms, larms, rlegs, llegs, head

    def process_csv(self, base_path='output', thresh=.3):
        sz = len(os.listdir(base_path)) // 3  # 3个文件一套

        candidates, subsets = [], []

        for file in range(sz):
            candidate_dat = pd.read_csv(os.path.join(
                base_path, f'{file}_candidate.csv'))
            subset_dat = pd.read_csv(os.path.join(
                base_path, f'{file}_subset.csv'))
            candidates.append(candidate_dat)
            subsets.append(subset_dat)

        return Predict.post_process((candidates, subsets), True, thresh=thresh)

    def process(self, imgs: list, output_path='output'):
        out_imgs = []
        coordinates = []
        for ind, img in enumerate(imgs):
            ret = Predict.predict(img)
            if ret is None: continue
            out_img = ret['data']

            out_imgs.append(out_img)
            coordinates.append(ret['coordinates'])

        return {'coordinates': coordinates, 'data': out_imgs}

    # TODO use faster video split library
    def split_video(self, video: str, output_path='./data', fps_limit=10) -> list:
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

    def dis(self, pos1, pos2):
        if np.isnan(pos1).any() or np.isnan(pos2).any():
            return 0
        ans = int(math.sqrt((pos1[0] - pos2[0]) **
                            2 + (pos1[1] - pos2[1]) ** 2))
        return ans

    def body_dis(self, body1, body2):
        ans = 0
        for idx, (name, point1) in enumerate(body1.items()):
            point2 = body2[name]
            ans += self.dis(point1, point2)
        return ans

    def dynamic_time_warping(self, series1, series2):
        l1, l2 = len(series1), len(series2)
        dp = [[0x7f7f7f7f for j in range(l2 + 1)] for i in range(l1 + 1)]
        dp[0][0] = 0

        for i in range(1, l1 + 1):
            for j in range(1, l2 + 1):
                cost = self.body_dis(series1[i - 1], series2[j - 1])
                dp[i][j] = cost + min(dp[i - 1][j - 1],
                                      dp[i - 1][j], dp[i][j - 1])
        return dp[l1][l2] / division

    @staticmethod
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
            for id in Predict.position:
                coordinates[Predict.position_map[id]] = (results.pose_landmarks.landmark[id].x * image_width,
                                                         results.pose_landmarks.landmark[id].y * image_height,
                                                         results.pose_landmarks.landmark[id].visibility)

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
            Predict.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        results = {
            'coordinates': coordinates,
            'data': annotated_image
        }
        return results

    def make_video(self, imgs, output_path, filename):
        assert len(imgs) > 0, 'the images count is zero!!!'

        img = imgs[0]
        imgInfo = img.shape
        size = (imgInfo[1], imgInfo[0])  # 宽高

        fps = 5  # 视频每秒1帧
        print(os.path.join(output_path + f'{filename[:-4]}_e.mp4'))
        video = cv.VideoWriter(os.path.join(output_path, f'{filename[:-4]}_e.mp4'), cv.VideoWriter_fourcc(*'H264'), fps,
                               size)
        for i in range(len(imgs)):
            img = imgs[i]
            video.write(img)

        video.release()

    @staticmethod
    def draw_landmarks(image, landmark_list, connections: Optional[List[Tuple[int, int]]] = None
                       , landmark_drawing_spec: Union[DrawingSpec,
                                                      Mapping[int, DrawingSpec]] = DrawingSpec(
                color=RED_COLOR),
                       connection_drawing_spec: Union[DrawingSpec,
                                                      Mapping[Tuple[int, int],
                                                              DrawingSpec]] = DrawingSpec()):

        if not landmark_list:
            return
        if image.shape[2] != _RGB_CHANNELS:
            raise ValueError('Input image must contain three channel rgb data.')
        image_rows, image_cols, _ = image.shape
        idx_to_coordinates = {}
        for idx, landmark in enumerate(landmark_list.landmark):
            if idx not in Predict.draw_position: continue
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

        pass

from __future__ import division

import copy
import math
import os
import time
from typing import Union

import cv2 as cv
import numpy as np
import pandas as pd
from django.contrib.auth import authenticate
from django.contrib.auth.hashers import make_password
from django.db import IntegrityError
from django.http import HttpResponse, JsonResponse
# deal with videos
from imutils.video import FileVideoStream
from rest_framework import views
from rest_framework.authtoken.models import Token
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.decorators import api_view
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.views import APIView

from backend.models import User, CustomUser

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

division = 10000


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
        """
        这里根据用户id，每个用户id创建一个目录，并且在用户目录里面根据课程创建一个目录，视频存到课程目录里面。
        # 如果考虑次数，则可能需要对视频重命名。
        """
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


class Predict(APIView):

    position = range(11, 33)

    position_map = {
        0: 'nose',
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
        32: 'Right foot index'
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def transform(data, center):
        for idx, (name, point) in enumerate(data.items()):
            print(f'center={center} and point={point}', end=' ')
            if point.any():
                point -= center
            print(f'center={center} and point={point}')

    def post(self, request):
        course = request.POST.get('course')
        userID = request.POST.get('userID')
        filename = request.POST.get('filename')

        uc_path = os.path.join('media', userID, course)
        split_path = os.path.join(uc_path, 'test_split')
        uv_path = os.path.join(uc_path, filename)
        pro_path = os.path.join(uc_path, 'pro')

        imgs = self.split_video(uv_path, output_path=split_path, fps_limit=10)

        res = self.process(imgs, output_path=pro_path)

        # try:
        #     self.make_video(res[2], uc_path, filename)
        # except AttributeError:
        #     return JsonResponse({'status': 403})

        # out = Predict.criterion_post_process(res[0], res[1])
        # criterion = self.process_csv(
        #     base_path=os.path.join('course', course, 'pro'))
        # cost_head = self.dynamic_time_warping(criterion[0], out[0])
        # cost_rarm = self.dynamic_time_warping(criterion[1], out[1])
        # cost_larm = self.dynamic_time_warping(criterion[2], out[2])
        # cost_rleg = self.dynamic_time_warping(criterion[3], out[3])
        # cost_lleg = self.dynamic_time_warping(criterion[4], out[4])

        # print('head', cost_head)
        # print('rarm', cost_rarm)
        # print('larm', cost_larm)
        # print('rleg', cost_rleg)
        # print('lleg', cost_lleg)
        # score = 100 - max(cost_head, cost_larm,
        #                   cost_larm, cost_rleg, cost_lleg)

        key = 50
        evaluate = ""
        # error = ["您的头部动作不标准 ", "您的右臂动作不标准 ",
        #        "您的左臂动作不标准 ", "您的右腿动作不标准 ", "您的左腿动作不标准 "]
        # if cost_head >= key:
        #     evaluate = evaluate + error[0]
        # if cost_rarm >= key:
        #     evaluate = evaluate + error[1]
        # if cost_larm >= key:
        #     evaluate = evaluate + error[2]
        # if cost_rleg >= key:
        #     evaluate = evaluate + error[3]
        # if cost_lleg >= key:
        #     evaluate = evaluate + error[4]

        if evaluate == "":
            evaluate = "您的动作非常完美"

        filename = filename[:-4] + "_e.mp4"
        return JsonResponse(data={'status': 204,
                                  'evaluate': evaluate,
                                  'url': f'{userID}/{course}/{filename}'})

    def delete_all_files(self, dir: str):
        for file in os.listdir(dir):
            os.remove(os.path.join(dir, file))

    def save_csv(self, data, filename, base_dir='./work'):
        dat = pd.DataFrame(data)
        dat.index = ['x', 'y']
        dat.to_csv(os.path.join(base_dir, filename))

    @staticmethod
    def post_process(candidates, subsets, thresh=0.3):
        heads = []
        rarms = []
        larms = []
        rlegs = []
        llegs = []

        pos_name = ('nose', 'neck',
                    'rshoulder', 'relbow', 'rhand',
                    'lshoulder', 'lelbow', 'lhand',
                    'rhip', 'rknee', 'rankle',
                    'lhip', 'lknee', 'lankle')

        assert len(candidates) == len(
            subsets), 'candidates length is not equal to that of subsets!'

        for i in range(len(candidates)):
            candidate_dat = candidates[i]
            subset_dat = subsets[i]

            pos = {}

            # 只需要看到14个就够了，不需要脸部数据
            for idx in Predict.position:
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

            heads.append({'nose': pos.get('nose'), 'neck': pos.get('neck')})
            rarms.append({'rshoulder': pos.get('rshoulder'), 'relbow': pos.get(
                'relbow'), 'rhand': pos.get('rhand')})
            larms.append({'lshoulder': pos.get('lshoulder'), 'lelbow': pos.get(
                'lelbow'), 'lhand': pos.get('lhand')})
            rlegs.append({'rhip': pos.get('rhip'), 'rknee': pos.get(
                'rknee'), 'rankle': pos.get('rankle')})
            llegs.append({'lhip': pos.get('lhip'), 'lknee': pos.get(
                'lknee'), 'lankle': pos.get('lankle')})
        return heads, rarms, larms, rlegs, llegs


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
        return Predict.post_process(candidates, subsets, thresh=thresh)

    def process(self, imgs: list, output_path='output'):
        out_imgs = []
        for ind, img in enumerate(imgs):
            ret = Predict.predict(img)
            out_img = ret['data']

            out_imgs.append(out_img)

        return ret['cordinates'], out_imgs

    # TODO use faster video split library
    def split_video(self, video: str, output_path='./data', fps_limit=10) -> list:
        """
        split a video to frames. return a list of a quantity of images
        use CamGear library
        """

        stream = FileVideoStream(video).start()

        imgs = []

        elapsed = 1 / fps_limit
        prev = time.time()
        i = 0
        print('start process video')
        while 1:
            # grab the frame from the threaded video file stream
            frame = stream.read()

            if frame is None:
                break

            elapsed_time = time.time() - prev
            if elapsed_time > elapsed:
                prev = time.time()
                imgs.append(frame)
                i += 1
        stream.stop()
        print('end process video')
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
    def predict(imgs: Union[str, np.ndarray]):

        cordinates = []
        out_imgs = []
        BG_COLOR = (192, 192, 192)

        with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True,
                min_detection_confidence=0.5) as pose:
            for idx, image in enumerate(imgs):
                cordinate = {}
                image_height, image_width, _ = image.shape
                # Convert the BGR image to RGB before processing.
                results = pose.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))

                if not results.pose_landmarks:
                    continue
                for id in Predict.position:
                    print(
                        f'{Predict.position_map[id]} coordinates: ('
                        f'{results.pose_landmarks.landmark[id].x * image_width},'
                        f'{results.pose_landmarks.landmark[id].y * image_height})'
                        f' {results.pose_landmarks.landmark[id].visibility}'
                    )
                    cordinate[Predict.position_map[id]] = (results.pose_landmarks.landmark[id].x * image_width,
                                                           results.pose_landmarks.landmark[id].y * image_height, results.pose_landmarks.landmark[id].visibility)

                annotated_image = image.copy()
                # Draw segmentation on the image.
                # To improve segmentation around boundaries, consider applying a joint
                # bilateral filter to "results.segmentation_mask" with "image".
                condition = np.stack(
                    (results.segmentation_mask,) * 3, axis=-1) > 0.1
                bg_image = np.zeros(image.shape, dtype=np.uint8)
                bg_image[:] = BG_COLOR
                annotated_image = np.where(
                    condition, annotated_image, bg_image)
                # Draw pose landmarks on the image.
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                cordinates.append(cordinate)
                out_imgs.append(annotated_image)
        results = {
            'cordinates': cordinate,
            'data': annotated_image}

        return results

    def make_video(self, imgs, output_path, filename):
        assert len(imgs) > 0, 'the images count is zero!!!'

        img = imgs[0]
        imgInfo = img.shape
        size = (imgInfo[1], imgInfo[0])  # 宽高

        fps = 1  # 视频每秒1帧
        video = cv.VideoWriter(output_path + f'\\{filename[:-4]}_e.mp4', cv.VideoWriter_fourcc(*'H264'), fps,
                               size)
        for i in range(len(imgs)):
            img = imgs[i]
            video.write(img)

        video.release()

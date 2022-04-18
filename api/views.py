from __future__ import division

import copy
import os
import time
from typing import Union

import cv2 as cv
import numpy as np
import paddle
import paddlehub as hub
import pandas as pd
from django.contrib.auth import authenticate
from django.contrib.auth.hashers import make_password
from django.db import IntegrityError
from django.http import HttpResponse, JsonResponse
from rest_framework import views
from rest_framework.authtoken.models import Token
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.decorators import api_view
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.views import APIView

from backend.models import User, CustomUser

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


# predict and process
class Predict(APIView):
    paddle.set_device(paddle.device.get_device())
    model = hub.Module(name='openpose_body_estimation')
    model.eval()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def post(self, request):
        course = request.POST.get('course')
        userID = request.POST.get('userID')
        filename = request.POST.get('filename')

        uc_path = os.path.join('media', userID, course)
        split_path = os.path.join(uc_path, 'test_split')
        uv_path = os.path.join(uc_path, filename)
        pro_path = os.path.join(uc_path, 'pro')

        self.split_video(uv_path, output_path=split_path)

        self.process(split_path, output_path=pro_path)

        try:
            self.make_video(pro_path, uc_path, len(
                os.listdir(pro_path)) // 3, filename)
        except AttributeError:
            return JsonResponse({'status': 403})

        out = self.process_csv(
            base_path=pro_path, output_path=os.path.join(uc_path, 'pro_csv'))
        criterion = self.process_csv(base_path=os.path.join('course', course, 'pro'),
                                     output_path=os.path.join('course', course, 'pro_csv'))
        cost_head = self.dynamic_time_warping(criterion[0], out[0])
        cost_rarm = self.dynamic_time_warping(criterion[1], out[1])
        cost_larm = self.dynamic_time_warping(criterion[2], out[2])
        cost_rleg = self.dynamic_time_warping(criterion[3], out[3])
        cost_lleg = self.dynamic_time_warping(criterion[4], out[4])

        print('head', cost_head)
        print('rarm', cost_rarm)
        print('larm', cost_larm)
        print('rleg', cost_rleg)
        print('lleg', cost_lleg)
        score = 100 - max(cost_head, cost_larm,
                          cost_larm, cost_rleg, cost_lleg)

        key = 10
        evaluate = ""
        str = ["您的头部动作不标准 ", "您的右臂动作不标准 ",
               "您的左臂动作不标准 ", "您的右腿动作不标准 ", "您的左腿动作不标准 "]
        if cost_head >= key:
            evaluate = evaluate + str[0]
        if cost_rarm >= key:
            evaluate = evaluate + str[1]
        if cost_larm >= key:
            evaluate = evaluate + str[2]
        if cost_rleg >= key:
            evaluate = evaluate + str[3]
        if cost_lleg >= key:
            evaluate = evaluate + str[4]

        if evaluate == "": evaluate = "您的动作非常完美"

        filename = filename[:-4] + "_e.mp4"
        return JsonResponse(data={'status': 204,
                                  'evaluate': evaluate,
                                  'url': f'{userID}/{course}/{filename}'})

    def transform(self, data, center):
        for idx, (name, point) in enumerate(data.items()):
            if point.any():
                point -= center

    def delete_all_files(self, dir: str):
        for file in os.listdir(dir):
            os.remove(os.path.join(dir, file))

    def save_csv(self, data, filename, base_dir='./work'):
        dat = pd.DataFrame(data)
        dat.index = ['x', 'y']
        dat.to_csv(os.path.join(base_dir, filename))

    def process_csv(self, base_path='output', thresh=.3, output_path='csv'):
        sz = len(os.listdir(base_path)) // 3  # 3个文件一套
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        else:
            self.delete_all_files(output_path)  # 删除先前所有文件

        series = []
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

        for file in range(sz):
            candidate_dat = pd.read_csv(os.path.join(
                base_path, f'{file}_candidate.csv'))
            subset_dat = pd.read_csv(os.path.join(
                base_path, f'{file}_subset.csv'))
            # print(subset_dat.shape)
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
                        [int(candidate_dat.loc[k][1]), int(candidate_dat.loc[k][2])])
            self.transform(pos, pos['neck'].copy())
            self.save_csv(pos, filename=f'{file}.csv', base_dir=output_path)
            series.append(pos)
            heads.append({'nose': pos.get('nose'), 'neck': pos.get('neck')})
            rarms.append({'rshoulder': pos.get('rshoulder'), 'relbow': pos.get(
                'relbow'), 'rhand': pos.get('rhand')})
            larms.append({'lshoulder': pos.get('lshoulder'), 'lelbow': pos.get(
                'lelbow'), 'lhand': pos.get('lhand')})
            rlegs.append({'rhip': pos.get('rhip'), 'rknee': pos.get(
                'rknee'), 'rankle': pos.get('rankle')})
            llegs.append({'lhip': pos.get('lhip'), 'lknee': pos.get(
                'lknee'), 'lankle': pos.get('lankle')})
        return heads, rarms, larms, rlegs, llegs  # 返回元组，方便后期索引

    def process(self, base_path='./data', output_path='output'):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        else:
            self.delete_all_files(output_path)
        for file in os.listdir(base_path):
            img = cv.imread(os.path.join(base_path, file))
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            ret = self.predict(img)
            out_img = ret['data']
            out_img = cv.cvtColor(out_img, cv.COLOR_RGB2BGR)
            cv.imwrite(os.path.join(
                output_path, f'{file[:-4]}_res.jpg'), out_img)
            candidate = ret['candidate']
            subset = ret['subset']
            candidate_dat = pd.DataFrame(candidate)
            subset_dat = pd.DataFrame(subset)
            candidate_dat.to_csv(os.path.join(
                output_path, f'{file[:-4]}_candidate.csv'))
            subset_dat.to_csv(os.path.join(
                output_path, f'{file[:-4]}_subset.csv'))

    # TODO use faster video split library
    def split_video(self, video: str, output_path='./data', fps_limit=10) -> bool:
        """
        split a video to frames. return a list of a quantity of images
        use opencv
        """
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        self.delete_all_files(output_path)

        cap = cv.VideoCapture(video)
        elapsed = 1 / fps_limit
        if cap.isOpened():
            prev = time.time()
            i = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                elapsed_time = time.time() - prev
                if elapsed_time > elapsed:
                    prev = time.time()
                    cv.imwrite(os.path.join(output_path, f'{i}.jpg'), frame)
                    i += 1
            return True
        return False

    def dis(self, pos1, pos2):
        if np.isnan(pos1).any() or np.isnan(pos2).any():
            return 0
        ans = (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2
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
    def predict(img: Union[str, np.ndarray]):
        if isinstance(img, str):
            orgImg = cv.imread(img)
        else:
            orgImg = img
        data, imageToTest_padded, pad = Predict.model.transform(orgImg)
        Mconv7_stage6_L1, Mconv7_stage6_L2 = Predict.model.forward(
            paddle.to_tensor(data))
        Mconv7_stage6_L1 = Mconv7_stage6_L1.numpy()
        Mconv7_stage6_L2 = Mconv7_stage6_L2.numpy()

        heatmap_avg = Predict.model.remove_pad(
            Mconv7_stage6_L2, imageToTest_padded, orgImg, pad)
        paf_avg = Predict.model.remove_pad(
            Mconv7_stage6_L1, imageToTest_padded, orgImg, pad)

        all_peaks = Predict.model.get_peak(heatmap_avg)
        connection_all, special_k = Predict.model.get_connection(
            all_peaks, paf_avg, orgImg)
        candidate, subset = Predict.model.get_candidate(
            all_peaks, connection_all, special_k)

        canvas = copy.deepcopy(orgImg)
        canvas = Predict.model.draw_pose(canvas, candidate, subset)

        results = {
            'candidate': candidate,
            'subset': subset,
            'data': canvas}

        return results

    def make_video(self, base_path, output_path, count, filename):

        img = cv.imread(os.path.join(base_path, '0_res.jpg'))
        imgInfo = img.shape
        size = (imgInfo[1], imgInfo[0])  # 宽高

        fps = 10  # 视频每秒1帧
        video = cv.VideoWriter(output_path + f'\\{filename[:-4]}_e.mp4', cv.VideoWriter_fourcc(*'H264'), fps,
                               size)
        for i in range(0, count):
            fileName = os.path.join(base_path, str(i) + '_res.jpg')
            img = cv.imread(fileName)  # 写入参数，参数是图片编码之前的数据
            video.write(img)

        video.release()
        cv.destroyAllWindows()

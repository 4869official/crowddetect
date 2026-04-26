# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import datetime
from datetime import datetime
import os
import json
import random
import threading
import time
import csv
import requests
import yaml
import glob
import cv2
import numpy as np
import math
import paddle
import sys
import copy
from copy import deepcopy
import openvino.runtime as ov
import sys
from types import SimpleNamespace
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt

from PIL import Image

from paddleocr import PaddleOCR, draw_ocr

# from collections import Sequence
from collections import defaultdict

from matplotlib import pyplot as plt

# from pytesseract import pytesseract

# add deploy path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)
sys.path.insert(0, os.path.dirname(__file__))
'''pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
'''
from openvino_infer.CSRNet_infer import CSRNet_predictor
from cfg_utils import argsparser, print_arguments, merge_cfg
from pipe_utils import PipeTimer
from pipe_utils import get_test_images, crop_image_with_det, crop_image_with_mot, parse_mot_res, parse_mot_keypoint
from pipe_utils import PushStream
from datacollector import DataCollector, Result

from python.infer import Detector, DetectorPicoDet
from python.keypoint_infer import KeyPointDetector
from python.keypoint_postprocess import translate_to_ori_images
from python.preprocess import decode_image, ShortSizeScale
from python.visualize import visualize_box_mask, visualize_attr, visualize_pose, visualize_action, \
    visualize_vehicleplate

from pptracking.python.mot_sde_infer import SDE_Detector
from pptracking.python.mot.visualize import plot_tracking_dict
from pptracking.python.mot.utils import flow_statistic, update_object_info

from pphuman.attr_infer import AttrDetector
from pphuman.video_action_infer import VideoActionRecognizer
from pphuman.action_infer import SkeletonActionRecognizer, DetActionRecognizer, ClsActionRecognizer
from pphuman.action_utils import KeyPointBuff, ActionVisualHelper
from pphuman.reid import ReID
from pphuman.mtmct import mtmct_process

from ppvehicle.vehicle_plate import PlateRecognizer
from ppvehicle.vehicle_attr import VehicleAttr

from download import auto_download_model
from model import Model

from dbmodel.models import crowdinfo, warning, address

import sys
import argparse
import closimi3
import torch.backends.cudnn as cudnn

from models.experimental import *
from utils1.datasets import *
from utils1.general import *
from utils1 import torch_utils

import gc

setlocal=1



def extract_clothing_histogram(image, color_space='hsv'):
    """
    提取整张图片的颜色直方图
    :param image: 输入图像
    :param color_space: 颜色空间 ('rgb' 或 'hsv')
    :return: 颜色直方图
    """
    if color_space == 'hsv':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def compare_histograms(hist1, hist2, method=cv2.HISTCMP_CORREL):
    """
    比较两个颜色直方图的相似性
    :param hist1: 第一个直方图
    :param hist2: 第二个直方图
    :param method: 比较方法 (默认使用相关系数)
    :return: 相似性得分
    """
    return cv2.compareHist(hist1, hist2, method)



class Smoke_File_Detector():
    def __init__(self):
        default_values = {
            'weights': [
                'C:\\Users\\19025\\Desktop\\crowd_vis-mainhbver\\crowd_vis-main\\pp-human\\pipeline\\weights\\smoke.pt'],
            'source': 'inference/images',
            'img_size': 640,
            'conf_thres': 0.25,
            'iou_thres': 0.45,
            'device': 'cpu',
            'classes': None,  # 由于 nargs='+'，如果没有提供，默认应该是 None 或者空列表 []
            'agnostic_nms': False,
            'augment': False
        }

        # 将字典转换为 SimpleNamespace 对象
        self.opt = SimpleNamespace(**default_values)

        self.device = torch_utils.select_device(self.opt.device)

        # Initialize
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load models
        self.model = attempt_load(self.opt.weights, map_location=self.device)
        self.imgsz = check_img_size(self.opt.img_size, s=self.model.stride.max())
        if self.half:
            self.model.half()

    # 本地调用
    def detect_test(self, test_list):
        for i, img in enumerate(test_list):
            im0 = img
            img = letterbox(img, new_shape=self.opt.img_size)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)  # faster

            # Run inference
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            if i == 0:
                batch_img = img
            else:
                batch_img = torch.cat([batch_img, img], axis=0)

        pred = self.model(batch_img, augment=self.opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                   agnostic=self.opt.agnostic_nms)

        # Process detections
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        batch_results = []
        for i, det in enumerate(pred):  # detections per image
            results = []
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(batch_img.shape[2:], det[:, :4], im0.shape).round()
                det = det.data.cpu().numpy()
                for *xyxy, conf, cls in det:
                    w = xyxy[2] - xyxy[0]
                    h = xyxy[3] - xyxy[1]
                    result = {'bbox': xyxy, 'label': names[int(cls)], 'conf': conf}
                    results.append(result)

            batch_results.append(results)

        # print(batch_results)
        return batch_results

    # server调用
    def detect(self, **kwargs):
        params = kwargs
        test_list = [params["img"]]
        for i, img in enumerate(test_list):
            im0 = img
            img = letterbox(img, new_shape=self.opt.img_size)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            # Run inference
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            if i == 0:
                batch_img = img
            else:
                batch_img = torch.cat([batch_img, img], axis=0)

        pred = self.model(batch_img, augment=self.opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                   agnostic=self.opt.agnostic_nms)

        # Process detections
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        batch_results = []
        for i, det in enumerate(pred):  # detections per image
            results = []
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(batch_img.shape[2:], det[:, :4], im0.shape).round()
                det = det.cuda().data.cpu().numpy()
                for *xyxy, conf, cls in det:
                    w = xyxy[2] - xyxy[0]
                    h = xyxy[3] - xyxy[1]
                    result = {'bbox': xyxy, 'label': names[int(cls)], 'conf': conf}
                    results.append(result)

            batch_results.append(results)

        # print(batch_results)
        return batch_results


# def gettextinfo(text):

def getinfo(caid):
    # 假设CSV文件的路径
    csv_file_path = 'C:\\Users\\19025\\Desktop\\crowd_vis-mainhbver\\crowd_vis-main\\pp-human\\pipeline\\govinf.csv'

    # 打开CSV文件
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        # 创建CSV阅读器
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        print(fieldnames)

        # 遍历CSV文件的每一行
        for row in reader:
            # 假设CSV文件中有列名为'name'和'phone'
            print(row)
            print(row['唯一标识码'])
            print(caid)
            if row['唯一标识码'] == caid:  # 将'name'替换为你CSV文件中对应的列名
                print(row['唯一标识码'] == caid)
                print(row['唯一标识码'])
                print(caid)
                qwe = row['\ufeff通道名称']
                asd = row['经度']
                zxc = row['纬度']
                rty = row['类别']
                print(qwe)
                print(asd)
                print(zxc)

                return qwe, asd, zxc, rty  # 将'phone'替换为你CSV文件中对应的列名


def calculate_complexity_similarity(img1str, img2str):
    # 加载两张图片
    img1 = cv2.imread(img1str)
    img2 = cv2.imread(img2str)

    # 将图片转换为灰度图像
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 创建ORB特征检测器
    orb = cv2.ORB_create()

    # 检测特征点和描述符
    kp1, des1 = orb.detectAndCompute(gray_img1, None)
    kp2, des2 = orb.detectAndCompute(gray_img2, None)

    # 创建暴力匹配器
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 进行特征匹配
    matches = bf.match(des1, des2)
    similarity = 0.0
    # 根据特征点匹配结果计算相似度
    if len(matches) > 0:
        similarity = sum([match.distance for match in matches]) / len(matches)
        print('图片相似度为：', similarity)
    else:
        print('未找到匹配的特征点')
        # 调用函数进行图片相似度计算,计算简单的图片相似度
    return similarity


def calculate_histogram_similarity(img1_path, img2_path):
    # 读取两张图片
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # 计算直方图
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])

    # 归一化直方图
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

    # 比较直方图
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    print(similarity)
    # 30%的几率是那应该不是一个东西
    '''

    if similarity < 0.3:
        return similarity
    if similarity < 0.6:
        similarity = calculate_complexity_similarity(img1_path, img2_path)

    '''

    return similarity


class ImageWindow(QWidget):
    def __init__(self, frame_rgb, humannum, humannum2):
        super().__init__()

        # 设置窗口标题和初始大小
        self.setWindowTitle('PyQt5 Image Display')
        self.setGeometry(100, 100, 800, 800)

        humannum += " 预警人数："
        humannum += humannum2

        imagepathhh = "E:\\apache-tomcat-9.0.52\\webapps\\ROOT\\tupian\\"
        imagepath = imagepathhh + str(time.localtime()) + ".jpg"
        frame_rgb.save(imagepath)

        # 创建一个QLabel来显示图片
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(800, 500)
        self.image_label.move(0, 0)
        # self.image_label.setAlignment(Qt.AlignCenter)

        self.label = QLabel(self)
        self.label.setText(humannum)
        font = QFont("黑体", 18)
        self.label.setFont(font)
        self.label.setFixedSize(600, 100)
        self.label.move(50, 600)

        frame_rgb = QImage(frame_rgb)
        pixmap = QPixmap.fromImage(frame_rgb)  # 替换为你的图片路径

        # 将图片设置到QLabel上
        self.image_label.setPixmap(pixmap)

        # 创建一个垂直布局并添加QLabel
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        # 设置布局
        self.setLayout(layout)


def data_uploader(url, data):
    requests.post(url, json=data,
                  headers={'Content-Type': 'application/json;charset=UTF-8'})


class Pipeline(object):
    """
    Pipeline

    Args:
        args (argparse.Namespace): arguments in pipeline, which contains environment and runtime settings
        cfg (dict): config of models in pipeline
    """

    def __init__(self, args, cfg):
        self.multi_camera = False
        reid_cfg = cfg.get('REID', False)
        self.enable_mtmct = reid_cfg['enable'] if reid_cfg else False
        self.is_video = False
        self.output_dir = args.output_dir
        self.vis_result = cfg['visual']
        self.input = self._parse_input(args.image_file, args.image_dir,
                                       args.video_file, args.video_dir,
                                       args.camera_id, args.rtsp)
        self.input = 0
        print(666)
        if self.multi_camera:
            print(666)
            self.predictor = []
            for name in self.input:
                predictor_item = PipePredictor(
                    args, cfg, is_video=True, multi_camera=True)
                predictor_item.set_file_name(name)
                self.predictor.append(predictor_item)

        else:
            self.predictor = PipePredictor(args, cfg, self.is_video)
            print(666)
            if self.is_video:
                print(666)
                self.predictor.set_file_name(self.input)

    def _parse_input(self, image_file, image_dir, video_file, video_dir,
                     camera_id, rtsp):

        # parse input as is_video and multi_camera

        if image_file is not None or image_dir is not None:
            input = get_test_images(image_dir, image_file)
            self.is_video = False
            self.multi_camera = False
            print(666)

        elif video_file is not None:
            assert os.path.exists(
                video_file
            ) or 'rtsp' in video_file, "video_file not exists and not an rtsp site."
            self.multi_camera = False
            input = video_file
            self.is_video = True
            print(666)

        elif video_dir is not None:
            videof = [os.path.join(video_dir, x) for x in os.listdir(video_dir)]
            if len(videof) > 1:
                self.multi_camera = True
                videof.sort()
                input = videof
            else:
                input = videof[0]
            self.is_video = True
            print(666)

        elif rtsp is not None:
            if len(rtsp) > 1:
                rtsp = [rtsp_item for rtsp_item in rtsp if 'rtsp' in rtsp_item]
                self.multi_camera = True
                input = rtsp
            else:
                self.multi_camera = False
                input = rtsp[0]
            self.is_video = True
            print(666)

        elif camera_id != -1:
            self.multi_camera = False
            input = camera_id
            self.is_video = True
            print(66666)

        else:
            raise ValueError(
                "Illegal Input, please set one of ['video_file', 'camera_id', 'image_file', 'image_dir']"
            )
        print(666)

        return input

    def run_multithreads(self):
        import threading
        if self.multi_camera:
            multi_res = []
            threads = []
            for idx, (predictor,
                      input) in enumerate(zip(self.predictor, self.input)):
                thread = threading.Thread(
                    name=str(idx).zfill(3),
                    target=predictor.run,
                    args=(input, idx))
                threads.append(thread)

            for thread in threads:
                thread.start()

            for predictor, thread in zip(self.predictor, threads):
                thread.join()
                collector_data = predictor.get_result()
                multi_res.append(collector_data)

            if self.enable_mtmct:
                mtmct_process(
                    multi_res,
                    self.input,
                    mtmct_vis=self.vis_result,
                    output_dir=self.output_dir)

        else:
            self.predictor.run(self.input)

    def run(self):
        if self.multi_camera:
            multi_res = []
            for predictor, input in zip(self.predictor, self.input):
                predictor.run(input)
                collector_data = predictor.get_result()
                multi_res.append(collector_data)
            if self.enable_mtmct:
                mtmct_process(
                    multi_res,
                    self.input,
                    mtmct_vis=self.vis_result,
                    output_dir=self.output_dir)

        else:
            self.predictor.run(self.input)


def get_model_dir(cfg):
    """
        Auto download inference model if the model_path is a url link.
        Otherwise it will use the model_path directly.
    """
    for key in cfg.keys():
        if type(cfg[key]) == dict and \
                ("enable" in cfg[key].keys() and cfg[key]['enable']
                 or "enable" not in cfg[key].keys()):

            if "model_dir" in cfg[key].keys():
                model_dir = cfg[key]["model_dir"]
                downloaded_model_dir = auto_download_model(model_dir)
                if downloaded_model_dir:
                    model_dir = downloaded_model_dir
                    cfg[key]["model_dir"] = model_dir
                print(key, " model dir: ", model_dir)
            elif key == "VEHICLE_PLATE":
                det_model_dir = cfg[key]["det_model_dir"]
                downloaded_det_model_dir = auto_download_model(det_model_dir)
                if downloaded_det_model_dir:
                    det_model_dir = downloaded_det_model_dir
                    cfg[key]["det_model_dir"] = det_model_dir
                print("det_model_dir model dir: ", det_model_dir)

                rec_model_dir = cfg[key]["rec_model_dir"]
                downloaded_rec_model_dir = auto_download_model(rec_model_dir)
                if downloaded_rec_model_dir:
                    rec_model_dir = downloaded_rec_model_dir
                    cfg[key]["rec_model_dir"] = rec_model_dir
                print("rec_model_dir model dir: ", rec_model_dir)

        elif key == "MOT":  # for idbased and skeletonbased actions
            model_dir = cfg[key]["model_dir"]
            downloaded_model_dir = auto_download_model(model_dir)
            if downloaded_model_dir:
                model_dir = downloaded_model_dir
                cfg[key]["model_dir"] = model_dir
            print("mot_model_dir model_dir: ", model_dir)


def getid():
    aid = []
    csv_file_path = 'C:\\Users\\19025\\Desktop\\crowd_vis-mainhbver\\crowd_vis-main\\pp-human\\pipeline\\govinf.csv'

    # 打开CSV文件
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        # 创建CSV阅读器
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        print(fieldnames)

        # 遍历CSV文件的每一行
        for row in reader:
            print(row)
            aid.append(row[fieldnames[1]])
            # 假设CSV文件中有列名为'name'和'phone'
            print(aid)
        return aid


camera_ids = getid()

rtsplinks = [
    f"rtsp://13.1.4.60:8319/dss/monitor/param?cameraid={cameraid}&substream=1" for cameraid in camera_ids
]

import re


def find_numbers(incomplete_str, data):
    # 正则表达式，用于匹配字符串后面的数字
    # pattern = re.compile(re.escape(incomplete_str) + r'(\d+)\s+(\d+)')

    for itemdex in range(len(data)):
        item = data[itemdex]

        # 搜索匹配项
        # match = pattern.search(item)
        m1, m2, m3 = item.split()
        if m1 in incomplete_str:
            numo1 = m2  # 第一个数字
            numo2 = m3  # 第二个数字
            textgov = gov[itemdex]

            return numo1, numo2, textgov

    return None, None, ""  # 如果没有找到匹配项，返回None


# 示例数据
data = [
    "省委	114.530399	38.037707",
    "省人民 	114.530399	38.037707",
    "庄市人 	114.471399	38.045801",
    "高新区管	114.629760	37.978214",
    "高邑 	114.610582	37.614297",
    "长安 	114.539001	38.036805",
    "新华 	114.463904	38.050749",
    "裕华 	114.531599	38.007002",
    "藁城	114.846562	38.022177",
    "鹿泉	114.313592	38.086571",
    "栾城	114.647922	37.900915",
    "矿区 	115.035186	37.751923",
    "灵寿 	114.382499	38.308680",
    "行唐 	114.459516	38.485159",
    "赞皇 	114.386375	37.665205",
    "平山 	114.185988	38.260312",
    "新乐 	114.683522	38.342929",
    "深泽 	115.200890	38.184572",
    "井陉 	114.144548	38.033075",
    "正定 	114.570320	38.146515",
    "无极 	114.975443	38.179244",
    "元氏 	114.523326	37.768929",
    "赵县 	114.775913	37.756883",
    "承德市人民 	117.942997	40.973785",
    "双桥区 	117.943121	40.974679",
    "双滦区 	117.799570	40.958732",
    "鹰手营子矿 	117.798725	40.579440",
    "平泉 	118.702032	41.018482",
    "承德 	118.174019	40.768700",
    "兴隆 	117.500945	40.417374",
    "滦平 	117.332652	40.941644",
    "隆化 	117.739026	41.314402",
    "丰宁满族自治 	116.647614	41.209711",
    "双桥区 	117.943121	40.974679",
    "承德市围场满族 	117.758825	41.938349",
    "唐山市人民 	118.180545	39.631459",
    "山市路南区 	118.154348	39.624988",
    "山市路北区 	118.201085	39.625079",
    "古冶 	118.447134	39.733920",
    "开平 	118.262246	39.671634",
    "丰南 	118.084985	39.575885",
    "芦台 	117.600292	39.373116",
    "山市高新区	118.178852	39.676681",
    "曹妃甸 	118.460197	39.273528",
    "市海港	119.003620	39.232040",
    "滦南 	118.683630	39.520291",
    "乐亭 	118.912867	39.427562",
    "迁西 	118.315562	40.142381",
    "迁安 	118.701021	39.998861",
    "玉田 	117.738196	39.901827",
    "路遵化	118.078395	40.035606",
    "丰润区 	118.162426	39.832919",
    "滦州区 	118.703010	39.740963",
    "岛市人民 	119.520220	39.888243",
    "岛市海港 	119.564300	39.948462",
    "山海关 	119.775187	39.978882",
    "岛市北戴 	119.484490	39.834912",
    "昌黎 	119.196744	39.700812",
    "岛市抚宁区 	119.246655	39.876717",
    "岛市卢龙 	118.891931	39.892564",
    "青龙满 	118.951141	40.406624",
    "邯郸市人民 	114.540911	36.626238",
    "邯山区 	114.528248	36.593807",
    "丛台区区 	114.492875	36.636434",
    "复兴区 	114.462287	36.639086",
    "峰矿区 	114.212571	36.419298",
    "临漳 	114.585514	36.322504",
    "成安 	114.668864	36.444171",
    "郸市大 	115.148669	36.287259",
    "涉县 	113.692157	36.586073",
    "磁县 	114.373965	36.375070",
    "邱县 	115.200049	36.811783",
    "鸡泽县 	114.889061	36.910221",
    "广平 	114.921052	36.476497",
    "馆陶 	115.281287	36.547122",
    "魏县 	114.939217	36.359260",
    "曲周县 	114.956552	36.768038",
    "武安 	114.203058	36.696458",
    "肥乡 	114.801228	36.549835",
    "永年县 	114.536626	36.741880",
    "经开区 	114.666896	36.671481",
    "永年区 	114.544046	36.743626",
    "邢台市市委	114.496719	37.060387",
    "囊都区 	114.507443	37.071314",
    "信都区 	114.468338	37.094036",
    "仁泽区 	114.670497	37.130823",
    "南和区 	114.683313	37.012957",
    "内丘 	114.514043	37.287312",
    "临城 	114.498651	37.444512",
    "隆尧 	114.771552	37.350858",
    "柏乡 	114.695321	37.483162",
    "宁晋 	114.942731	37.624768",
    "巨鹿 	115.037490	37.220441",
    "平乡 	115.029980	37.063672",
    "新河 	115.251005	37.520891",
    "广宗 	115.141937	37.074242",
    "威县 	115.266829	36.975164",
    "临西 	115.500705	36.871618",
    "清河 	115.665042	37.046001",
    "南宫市 	115.408660	37.358907",
    "沙河市 	114.503023	36.855548",
    "保定市市委	115.493569	38.869611",
    "竞秀区 	115.458671	38.877318",
    "莲池区 	115.497179	38.883243",
    "涞水 	115.714709	39.393885",
    "阜平 	114.196328	38.848112",
    "定兴 	115.808503	39.262526",
    "唐县 	114.982968	38.748477",
    "高阳 	115.776279	38.700275",
    "涞源 	114.694525	39.359980",
    "望都 	115.155170	38.695827",
    "义县 	115.464523	38.874476",
    "曲阳 	114.740476	38.614409",
    "顺平 	115.133698	38.839373",
    "博野 	115.465168	38.457401",
    "涿州市 	115.973001	39.486808",
    "安国市 	115.328282	38.418240",
    "高碑店市 	115.873612	39.327233",
    "高新区 	115.473069	38.935479",
    "白沟 	116.034760	39.121266",
    "满城区 	115.322246	38.949732",
    "徐水区 	115.655772	39.018781",
    "清苑区 	115.489880	38.765270",
    "蠡县 	115.583701	38.488064",
    "家口市人民 	114.885895	40.768931",
    "口桥东区	114.894114	40.788472",
    "口桥西区	114.869991	40.818737",
    "宣化 	115.098012	40.609831",
    "下花园 	115.287127	40.502628",
    "万全 	114.740584	40.767377",
    "张北 	114.719471	41.158426",
    "康保 	114.612543	41.846269",
    "沽源 	115.688544	41.670497",
    "尚义 	113.968763	41.076588",
    "蔚县 	114.589136	39.840154",
    "阳原 	114.150267	40.104303",
    "怀安 	114.384383	40.674828",
    "怀来 	115.517868	40.415625",
    "涿鹿 	115.196744	40.382397",
    "赤城 	115.831256	40.913348",
    "崇礼区 	115.281995	40.974356",
    "察北区 	115.023155	41.497643",
    "经开区 	114.877652	40.772139",
    "沧州市人民 	116.838715	38.304676",
    "新华区区	116.866309	38.314094",
    "运河区 	116.842964	38.283456",
    "沧县 	116.877857	38.292950",
    "青县 	116.804137	38.583657",
    "东光 	116.537320	37.887753",
    "海兴 	117.497545	38.143308",
    "盐山 	117.230681	38.058074",
    "肃宁 	115.829619	38.423044",
    "南皮 	116.707633	38.039886",
    "吴桥 	116.390520	37.627264",
    "献县 	116.152058	38.180381",
    "孟村回 	117.104621	38.054122",
    "泊头 	116.580202	38.084471",
    "任丘 	116.084412	38.685325",
    "黄骅 	117.331223	38.373534",
    "河间 	116.088543	38.351927",
    "渤海新区 	117.757611	38.276459",
    "沧州市高新开发区 	117.351438	38.463803",
    "廊坊市人民 	116.683546	39.538304",
    "安次区 	116.703028	39.520831",
    "广阳区 	116.710667	39.523430",
    "固安 	116.301603	39.438796",
    "永清 	116.505147	39.330272",
    "香河 	117.006158	39.763923",
    "大城 	116.653318	38.703219",
    "文安 	116.459776	38.873919",
    "大厂回族 	116.988284	39.884674",
    "霸州 	116.388948	39.125327",
    "三河 	117.077834	39.982426",
    "市开发区 	116.746868	39.586680",
    "人民 	115.668987	37.739367",
    "桃城 	115.675208	37.735152",
    "冀州 	115.579392	37.550922",
    "枣强 	115.724365	37.514217",
    "武邑 	115.888627	37.803039",
    "武强 	115.982119	38.041447",
    "饶阳 	115.725647	38.235683",
    "安平 	115.519922	38.235072",
    "故城 	115.964308	37.350098",
    "景县 	116.270558	37.692831",
    "阜城 	116.175424	37.862984",
    "深州 	115.559757	38.001341",
    "辛集 	115.216023	37.945592",
    "定州 	114.989794	38.515601",
    "省冀中 	115.035186	37.751923"

]

gov = [
    "河北省省委",
    "河北省省政府",
    "石家庄市人民政府",
    "石家庄市高新区管委会",
    "石家庄市高邑县人民政府",
    "石家庄市长安区人民政府",
    "石家庄市新华区人民政府",
    "石家庄市裕华区人民政府",
    "石家庄市藁城区人民政府",
    "石家庄市鹿泉区人民政府",
    "石家庄市栾城区人民政府",
    "石家庄市矿区人民政府",
    "石家庄市灵寿县人民政府",
    "石家庄市行唐县人民政府",
    "石家庄市赞皇县人民政府",
    "石家庄市平山县人民政府",
    "石家庄市新乐市人民政府",
    "石家庄市深泽市人民政府",
    "石家庄市井陉县人民政府",
    "石家庄市正定县人民政府",
    "石家庄市无极县人民政府",
    "石家庄市元氏县人民政府",
    "石家庄市赵县人民政府",
    "承德市人民政府",
    "承德市双桥区人民政府",
    "承德市双滦区人民政府",
    "承德市鹰手营子矿区人民政府",
    "承德市平泉市人民政府",
    "承德市承德县人民政府",
    "承德市兴隆县人民政府",
    "承德市滦平县人民政府",
    "承德市隆化县人民政府",
    "承德市丰宁满族自治县人民政府",
    "承德市双桥区人民政府",
    "承德市围场满族蒙古族自治县人民政府",
    "唐山市人民政府",
    "唐山市路南区人民政府",
    "唐山市路北区人民政府",
    "唐山市古冶区人民政府",
    "唐山市开平区人民政府",
    "唐山市丰南区人民政府",
    "唐山市芦台区人民政府",
    "唐山市高新区管委会",
    "唐山市曹妃甸区人民政府",
    "唐山市海港经济开发区管委会",
    "唐山市滦南县人民政府",
    "唐山市乐亭县人民政府",
    "唐山市迁西县人民政府",
    "唐山市迁安市人民政府",
    "唐山市玉田县人民政府",
    "唐山市路遵化市人民政府",
    "唐山市丰润区人民政府",
    "唐山市滦州区人民政府",
    "秦皇岛市人民政府",
    "秦皇岛市海港区人民政府",
    "秦皇岛市山海关区人民政府",
    "秦皇岛市北戴河区人民政府",
    "秦皇岛市昌黎县人民政府",
    "秦皇岛市抚宁区人民政府",
    "秦皇岛市卢龙县人民政府",
    "秦皇岛市青龙满族自治县人民政府",
    "邯郸市人民政府",
    "邯郸市邯山区人民政府",
    "邯郸市丛台区区人民政府",
    "邯郸市复兴区人民政府",
    "邯郸市峰峰矿区人民政府",
    "邯郸市临漳县人民政府",
    "邯郸市成安县人民政府",
    "邯郸市大名县人民政府",
    "邯郸市涉县人民政府",
    "邯郸市磁县人民政府",
    "邯郸市邱县人民政府",
    "邯郸市鸡泽县人民政府",
    "邯郸市广平县人民政府",
    "邯郸市馆陶县人民政府",
    "邯郸市魏县人民政府",
    "邯郸市曲周县人民政府",
    "邯郸市武安市人民政府",
    "邯郸市肥乡县人民政府",
    "邯郸市永年县人民政府",
    "邯郸市经开区人民政府",
    "邯郸市永年区人民政府",
    "邢台市市委",
    "邢台市囊都区人民政府",
    "邢台市信都区人民政府",
    "邢台市仁泽区人民政府",
    "邢台市南和区人民政府",
    "邢台市内丘县人民政府",
    "邢台市临城县人民政府",
    "邢台市隆尧县人民政府",
    "邢台市柏乡县人民政府",
    "邢台市宁晋县人民政府",
    "邢台市巨鹿县人民政府",
    "邢台市平乡县人民政府",
    "邢台市新河县人民政府",
    "邢台市广宗县人民政府",
    "邢台市威县人民政府",
    "邢台市临西县人民政府",
    "邢台市清河县人民政府",
    "邢台市南宫市人民政府",
    "邢台市沙河市人民政府",
    "保定市市委",
    "保定市竞秀区人民政府",
    "保定市莲池区人民政府",
    "保定市涞水县人民政府",
    "保定市阜平县人民政府",
    "保定市定兴县人民政府",
    "保定市唐县人民政府",
    "保定市高阳县人民政府",
    "保定市涞源县人民政府",
    "保定市望都县人民政府",
    "保定市义县人民政府",
    "保定市曲阳县人民政府",
    "保定市顺平县人民政府",
    "保定市博野县人民政府",
    "保定市涿州市人民政府",
    "保定市安国市人民政府",
    "保定市高碑店市人民政府",
    "保定市高新区人民政府",
    "保定市白沟人民政府",
    "保定市满城区人民政府",
    "保定市徐水区人民政府",
    "保定市清苑区人民政府",
    "保定市蠡县人民政府",
    "张家口市人民政府",
    "张家口桥东区区委",
    "张家口桥西区区委",
    "张家口市宣化区人民政府",
    "张家口市下花园区人民政府",
    "张家口市万全区人民政府",
    "张家口市张北县人民政府",
    "张家口市康保县人民政府",
    "张家口市沽源县人民政府",
    "张家口市尚义县人民政府",
    "张家口市蔚县人民政府",
    "张家口市阳原县人民政府",
    "张家口市怀安县人民政府",
    "张家口市怀来县人民政府",
    "张家口市涿鹿县人民政府",
    "张家口市赤城县人民政府",
    "张家口市崇礼区人民政府",
    "张家口市察北区人民政府",
    "张家口市经开区人民政府",
    "沧州市人民政府",
    "沧州市新华区区委",
    "沧州市运河区人民政府",
    "沧州市沧县人民政府",
    "沧州市青县人民政府",
    "沧州市东光县人民政府",
    "沧州市海兴县人民政府",
    "沧州市盐山县人民政府",
    "沧州市肃宁县人民政府",
    "沧州市南皮县人民政府",
    "沧州市吴桥县人民政府",
    "沧州市献县人民政府",
    "沧州市孟村回族自治县人民政府",
    "沧州市泊头市人民政府",
    "沧州市任丘市人民政府",
    "沧州市黄骅市人民政府",
    "沧州市河间市人民政府",
    "沧州市渤海新区人民政府",
    "沧州市高新开发区人民政府",
    "廊坊市人民政府",
    "廊坊市安次区人民政府",
    "廊坊市广阳区人民政府",
    "廊坊市固安县人民政府",
    "廊坊市永清县人民政府",
    "廊坊市香河县人民政府",
    "廊坊市大城县人民政府",
    "廊坊市文安县人民政府",
    "廊坊市大厂回族自治县人民政府",
    "廊坊市霸州市人民政府",
    "廊坊市三河市人民政府",
    "廊坊市开发区人民政府",
    "衡水市人民政府",
    "衡水市桃城区人民政府",
    "衡水市冀州区人民政府",
    "衡水市枣强县人民政府",
    "衡水市武邑县人民政府",
    "衡水市武强县人民政府",
    "衡水市饶阳县人民政府",
    "衡水市安平县人民政府",
    "衡水市故城县人民政府",
    "衡水市景县人民政府",
    "衡水市阜城县人民政府",
    "衡水市深州市人民政府",
    "河北省辛集市人民政府",
    "河北省定州市人民政府",
    "河北省冀中人民政府"
]


class PipePredictor(object):
    """
    Predictor in single camera

    The pipeline for image input:

        1. Detection
        2. Detection -> Attribute

    The pipeline for video input:

        1. Tracking
        2. Tracking -> Attribute
        3. Tracking -> KeyPoint -> SkeletonAction Recognition
        4. VideoAction Recognition

    Args:
        args (argparse.Namespace): arguments in pipeline, which contains environment and runtime settings
        cfg (dict): config of models in pipeline
        is_video (bool): whether the input is video, default as False
        multi_camera (bool): whether to use multi camera in pipeline,
            default as False
    """

    def __init__(self, args, cfg, is_video=True, multi_camera=False):
        if args.run_mode == "openvino":
            self.core = ov.Core()
            print("OpenVINO Runtime Core Created!")
        else:
            self.core = None
        # general module for pphuman and ppvehicle
        self.with_mot = cfg.get('MOT', False)['enable'] if cfg.get(
            'MOT', False) else False
        self.with_human_attr = cfg.get('ATTR', False)['enable'] if cfg.get(
            'ATTR', False) else False
        if self.with_mot:
            print('Multi-Object Tracking enabled')
        if self.with_human_attr:
            print('Human Attribute Recognition enabled')

        # only for pphuman
        self.with_skeleton_action = cfg.get(
            'SKELETON_ACTION', False)['enable'] if cfg.get('SKELETON_ACTION',
                                                           False) else False
        self.with_video_action = cfg.get(
            'VIDEO_ACTION', False)['enable'] if cfg.get('VIDEO_ACTION',
                                                        False) else False
        self.with_idbased_detaction = cfg.get(
            'ID_BASED_DETACTION', False)['enable'] if cfg.get(
            'ID_BASED_DETACTION', False) else False
        self.with_idbased_clsaction = cfg.get(
            'ID_BASED_CLSACTION', False)['enable'] if cfg.get(
            'ID_BASED_CLSACTION', False) else False
        self.with_mtmct = cfg.get('REID', False)['enable'] if cfg.get(
            'REID', False) else False

        if self.with_skeleton_action:
            print('SkeletonAction Recognition enabled')
        if self.with_video_action:
            print('VideoAction Recognition enabled')
        if self.with_idbased_detaction:
            print('IDBASED Detection Action Recognition enabled')
        if self.with_idbased_clsaction:
            print('IDBASED Classification Action Recognition enabled')
        if self.with_mtmct:
            print("MTMCT enabled")

        # only for ppvehicle
        self.with_vehicleplate = cfg.get(
            'VEHICLE_PLATE', False)['enable'] if cfg.get('VEHICLE_PLATE',
                                                         False) else False
        if self.with_vehicleplate:
            print('Vehicle Plate Recognition enabled')

        self.with_vehicle_attr = cfg.get(
            'VEHICLE_ATTR', False)['enable'] if cfg.get('VEHICLE_ATTR',
                                                        False) else False
        if self.with_vehicle_attr:
            print('Vehicle Attribute Recognition enabled')

        self.modebase = {
            "framebased": False,
            "videobased": False,
            "idbased": False,
            "skeletonbased": False
        }

        self.basemode = {
            "MOT": "idbased",
            "ATTR": "idbased",
            "VIDEO_ACTION": "videobased",
            "SKELETON_ACTION": "skeletonbased",
            "ID_BASED_DETACTION": "idbased",
            "ID_BASED_CLSACTION": "idbased",
            "REID": "idbased",
            "VEHICLE_PLATE": "idbased",
            "VEHICLE_ATTR": "idbased",
        }

        self.is_video = is_video
        self.multi_camera = multi_camera
        self.cfg = cfg

        self.output_dir = args.output_dir
        self.draw_center_traj = args.draw_center_traj
        self.secs_interval = args.secs_interval
        self.do_entrance_counting = args.do_entrance_counting
        self.do_break_in_counting = args.do_break_in_counting
        self.region_type = args.region_type
        self.region_polygon = args.region_polygon
        self.illegal_parking_time = args.illegal_parking_time

        self.warmup_frame = self.cfg['warmup_frame']
        self.pipeline_res = Result()
        self.pipe_timer = PipeTimer()
        self.file_name = None
        self.collector = DataCollector()
        self.heatmap = args.heatmap
        self.pushurl = args.pushurl

        # auto download inference model
        get_model_dir(self.cfg)

        if self.heatmap:
            self.heatmap_detector = CSRNet_predictor(os.path.join(os.path.dirname(__file__),
                                                                  "model",
                                                                  "CSRNet",
                                                                  "model.onnx"),
                                                     self.core)

        if self.with_vehicleplate:
            vehicleplate_cfg = self.cfg['VEHICLE_PLATE']
            self.vehicleplate_detector = PlateRecognizer(args, vehicleplate_cfg)
            basemode = self.basemode['VEHICLE_PLATE']
            self.modebase[basemode] = True

        if self.with_human_attr:
            attr_cfg = self.cfg['ATTR']
            basemode = self.basemode['ATTR']
            self.modebase[basemode] = True
            self.attr_predictor = AttrDetector.init_with_cfg(args, attr_cfg, core=self.core)

        if self.with_vehicle_attr:
            vehicleattr_cfg = self.cfg['VEHICLE_ATTR']
            basemode = self.basemode['VEHICLE_ATTR']
            self.modebase[basemode] = True
            self.vehicle_attr_predictor = VehicleAttr.init_with_cfg(
                args, vehicleattr_cfg)

        if not is_video:
            det_cfg = self.cfg['DET']
            model_dir = det_cfg['model_dir']
            batch_size = det_cfg['batch_size']
            self.det_predictor = Detector(
                model_dir, args.device, args.run_mode, batch_size,
                args.trt_min_shape, args.trt_max_shape, args.trt_opt_shape,
                args.trt_calib_mode, args.cpu_threads, args.enable_mkldnn)
        else:
            if self.with_idbased_detaction:
                idbased_detaction_cfg = self.cfg['ID_BASED_DETACTION']
                basemode = self.basemode['ID_BASED_DETACTION']
                self.modebase[basemode] = True

                self.det_action_predictor = DetActionRecognizer.init_with_cfg(
                    args, idbased_detaction_cfg, core=self.core)
                self.det_action_visual_helper = ActionVisualHelper(1)

            if self.with_idbased_clsaction:
                idbased_clsaction_cfg = self.cfg['ID_BASED_CLSACTION']
                basemode = self.basemode['ID_BASED_CLSACTION']
                self.modebase[basemode] = True

                self.cls_action_predictor = ClsActionRecognizer.init_with_cfg(
                    args, idbased_clsaction_cfg, core=self.core)
                self.cls_action_visual_helper = ActionVisualHelper(1)

            if self.with_skeleton_action:
                skeleton_action_cfg = self.cfg['SKELETON_ACTION']
                display_frames = skeleton_action_cfg['display_frames']
                self.coord_size = skeleton_action_cfg['coord_size']
                basemode = self.basemode['SKELETON_ACTION']
                self.modebase[basemode] = True
                skeleton_action_frames = skeleton_action_cfg['max_frames']

                self.skeleton_action_predictor = SkeletonActionRecognizer.init_with_cfg(
                    args, skeleton_action_cfg, core=self.core)
                self.skeleton_action_visual_helper = ActionVisualHelper(
                    display_frames)

                kpt_cfg = self.cfg['KPT']
                kpt_model_dir = kpt_cfg['model_dir']
                kpt_batch_size = kpt_cfg['batch_size']
                self.kpt_predictor = KeyPointDetector(
                    kpt_model_dir,
                    args.device,
                    args.run_mode,
                    kpt_batch_size,
                    args.trt_min_shape,
                    args.trt_max_shape,
                    args.trt_opt_shape,
                    args.trt_calib_mode,
                    args.cpu_threads,
                    args.enable_mkldnn,
                    use_dark=False,
                    core=self.core)
                self.kpt_buff = KeyPointBuff(skeleton_action_frames)

            if self.with_vehicleplate:
                vehicleplate_cfg = self.cfg['VEHICLE_PLATE']
                self.vehicleplate_detector = PlateRecognizer(args,
                                                             vehicleplate_cfg)
                basemode = self.basemode['VEHICLE_PLATE']
                self.modebase[basemode] = True

            if self.with_mtmct:
                reid_cfg = self.cfg['REID']
                basemode = self.basemode['REID']
                self.modebase[basemode] = True
                self.reid_predictor = ReID.init_with_cfg(args, reid_cfg)

            if self.with_mot or self.modebase["idbased"] or self.modebase[
                "skeletonbased"]:
                mot_cfg = self.cfg['MOT']
                model_dir = mot_cfg['model_dir']
                tracker_config = mot_cfg['tracker_config']
                batch_size = mot_cfg['batch_size']
                skip_frame_num = mot_cfg.get('skip_frame_num', -1)
                basemode = self.basemode['MOT']
                self.modebase[basemode] = True
                self.mot_predictor = SDE_Detector(
                    model_dir,
                    tracker_config,
                    args.device,
                    args.run_mode,
                    batch_size,
                    args.trt_min_shape,
                    args.trt_max_shape,
                    args.trt_opt_shape,
                    args.trt_calib_mode,
                    args.cpu_threads,
                    args.enable_mkldnn,
                    skip_frame_num=skip_frame_num,
                    draw_center_traj=self.draw_center_traj,
                    secs_interval=self.secs_interval,
                    do_entrance_counting=self.do_entrance_counting,
                    do_break_in_counting=self.do_break_in_counting,
                    region_type=self.region_type,
                    region_polygon=self.region_polygon,
                    core=self.core)

            if self.with_video_action:
                video_action_cfg = self.cfg['VIDEO_ACTION']
                basemode = self.basemode['VIDEO_ACTION']
                self.modebase[basemode] = True
                self.video_action_predictor = VideoActionRecognizer.init_with_cfg(
                    args, video_action_cfg, core=self.core)

    def set_file_name(self, path):
        if path is not None:
            self.file_name = os.path.split(path)[-1]
            if "." in self.file_name:
                self.file_name = self.file_name.split(".")[-2]
        else:
            # self.file_name = "output"
            # use camera id
            self.file_name = None

    def get_result(self):
        return self.collector.get_res()

    def run(self, input, thread_idx=0):
        if self.is_video:
            self.predict_video(input, thread_idx=thread_idx)
        else:
            self.predict_image(input)
        self.pipe_timer.info()

    def predict_image(self, input):
        # det
        # det -> attr
        print(666)
        batch_loop_cnt = math.ceil(
            float(len(input)) / self.det_predictor.batch_size)
        for i in range(batch_loop_cnt):
            start_index = i * self.det_predictor.batch_size
            end_index = min((i + 1) * self.det_predictor.batch_size, len(input))
            batch_file = input[start_index:end_index]
            batch_input = [decode_image(f, {})[0] for f in batch_file]

            if i > self.warmup_frame:
                self.pipe_timer.total_time.start()
                self.pipe_timer.module_time['det'].start()
            # det output format: class, score, xmin, ymin, xmax, ymax
            det_res = self.det_predictor.predict_image(
                batch_input, visual=False)
            det_res = self.det_predictor.filter_box(det_res,
                                                    self.cfg['crop_thresh'])
            # print(det_res[0][0][0][0][0][0][0][0])
            # print(det_res[3])
            if i > self.warmup_frame:
                self.pipe_timer.module_time['det'].end()
                self.pipe_timer.track_num += len(det_res['boxes'])
            self.pipeline_res.update(det_res, 'det')

            if self.with_human_attr:
                crop_inputs = crop_image_with_det(batch_input, det_res)
                attr_res_list = []

                if i > self.warmup_frame:
                    self.pipe_timer.module_time['attr'].start()

                for crop_input in crop_inputs:
                    attr_res = self.attr_predictor.predict_image(
                        crop_input, visual=False)
                    attr_res_list.extend(attr_res['output'])

                if i > self.warmup_frame:
                    self.pipe_timer.module_time['attr'].end()

                attr_res = {'output': attr_res_list}
                self.pipeline_res.update(attr_res, 'attr')

            if self.with_vehicle_attr:
                crop_inputs = crop_image_with_det(batch_input, det_res)
                vehicle_attr_res_list = []

                if i > self.warmup_frame:
                    self.pipe_timer.module_time['vehicle_attr'].start()

                for crop_input in crop_inputs:
                    attr_res = self.vehicle_attr_predictor.predict_image(
                        crop_input, visual=False)
                    vehicle_attr_res_list.extend(attr_res['output'])

                if i > self.warmup_frame:
                    self.pipe_timer.module_time['vehicle_attr'].end()

                attr_res = {'output': vehicle_attr_res_list}
                self.pipeline_res.update(attr_res, 'vehicle_attr')

            if self.with_vehicleplate:
                if i > self.warmup_frame:
                    self.pipe_timer.module_time['vehicleplate'].start()
                crop_inputs = crop_image_with_det(batch_input, det_res)
                platelicenses = []
                for crop_input in crop_inputs:
                    platelicense = self.vehicleplate_detector.get_platelicense(
                        crop_input)
                    platelicenses.extend(platelicense['plate'])
                if i > self.warmup_frame:
                    self.pipe_timer.module_time['vehicleplate'].end()
                vehicleplate_res = {'vehicleplate': platelicenses}
                self.pipeline_res.update(vehicleplate_res, 'vehicleplate')

            self.pipe_timer.img_num += len(batch_input)
            if i > self.warmup_frame:
                self.pipe_timer.total_time.end()

            if self.cfg['visual']:
                self.visualize_image(batch_file, batch_input, self.pipeline_res)

    def predict_video(self, video_file, thread_idx=0, return_im=False):
        # mot
        # mot -> attr
        # mot -> pose -> action
        print(999)
        # capture = cv2.VideoCapture(video_file)cv2.CAP_FFMPEG
        cameraindex = 0
        if setlocal==0:
            capture = cv2.VideoCapture(rtsplinks[cameraindex], cv2.CAP_FFMPEG)
        else:
            capture = cv2.VideoCapture(0)


        # Get Video info : resolution, fps, frame count
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        fps = 30
        print(fps)
        print(fps)
        print(fps)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("video fps: %d, frame_count: %d" % (fps, frame_count))

        if len(self.pushurl) > 0:
            video_out_name = 'output' if self.file_name is None else self.file_name
            pushurl = os.path.join(self.pushurl, video_out_name)
            print("the result will push stream to url:{}".format(pushurl))
            # pushstream = PushStream(pushurl)
            pushstream = PushStream()
            pushstream.initcmd(fps, width, height)
        elif self.cfg['visual']:
            video_out_name = 'output' if self.file_name is None else self.file_name
            if "rtsp" in video_file:
                video_out_name = video_out_name + "_t" + str(thread_idx).zfill(
                    2) + "_rtsp"
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            out_path = os.path.join(self.output_dir, time.strftime("%Y%m%d_%H%M%S", time.localtime()) + ".mp4")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # fourcc=-1
            writer = cv2.VideoWriter(
                os.path.join(self.output_dir, time.strftime("%Y%m%d_%H%M%S", time.localtime()) + ".mp4"), fourcc,
                fps - 45, (width, height))
            # writer = cv2.VideoWriter(os.path.join("C:\\Users\\ls\\Desktop\\output2", time.strftime("%Y%m%d_%H%M%S", time.localtime())+".mp4"), fourcc, fps-45, (width, height))
            writer.release()

        frame_id = 0

        entrance, records, center_traj = None, None, None
        if self.draw_center_traj:
            center_traj = [{}]
        id_set = set()
        interval_id_set = set()
        in_id_list = list()
        out_id_list = list()
        prev_center = dict()
        records = list()
        if self.do_entrance_counting or self.do_break_in_counting or self.illegal_parking_time != -1:
            if self.region_type == 'horizontal':
                entrance = [0, height / 2., width, height / 2.]
            elif self.region_type == 'vertical':
                entrance = [width / 2, 0., width / 2, height]
            elif self.region_type == 'custom':
                entrance = []
                assert len(
                    self.region_polygon
                ) % 2 == 0, "region_polygon should be pairs of coords points when do break_in counting."
                assert len(
                    self.region_polygon
                ) > 6, 'region_type is custom, region_polygon should be at least 3 pairs of point coords.'

                for i in range(0, len(self.region_polygon), 2):
                    entrance.append(
                        [self.region_polygon[i], self.region_polygon[i + 1]])
                entrance.append([width, height])
            else:
                raise ValueError("region_type:{} unsupported.".format(
                    self.region_type))

        video_fps = fps

        video_action_imgs = []

        if self.with_video_action:
            short_size = self.cfg["VIDEO_ACTION"]["short_size"]
            scale = ShortSizeScale(short_size)

        object_in_region_info = {
        }  # store info for vehicle parking in region
        illegal_parking_dict = None

        # app = QApplication(sys.argv)

        cont = True

        pppnum = 0

        # db_conn = DatabaseConnection()

        used = []

        camerastart = time.time()

        while (1):
            if time.time() - camerastart >= 6:
                if cameraindex <= len(rtsplinks) - 2:
                    cameraindex += 1
                else:
                    cameraindex = cameraindex - len(rtsplinks) + 1

                capture.release()  # 释放当前的视频捕获对象
                gc.collect()
                try:
                    if setlocal == 0:
                        capture = cv2.VideoCapture(rtsplinks[cameraindex], cv2.CAP_FFMPEG)
                        capture.set(cv2.CAP_PROP_BUFFERSIZE,5)
                    else:
                        capture = cv2.VideoCapture(0)
                except:
                    print("WA")
                camerastart = time.time()
            if frame_id % 10 == 0:
                print('Thread: {}; frame id: {}'.format(thread_idx, frame_id))

            ret, frame = capture.read()

            # writer.write(frame)
            while (not ret):
                if cameraindex <= len(rtsplinks) - 2:
                    cameraindex += 1
                else:
                    cameraindex = cameraindex - len(rtsplinks) + 1
                if setlocal == 0:
                    capture = cv2.VideoCapture(rtsplinks[cameraindex], cv2.CAP_FFMPEG)
                else:
                    capture = cv2.VideoCapture(0)
                ret, frame = capture.read()

            if not ret:
                break
            # cv2.imshow("frame",frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #cv2.imshow("ffffff", frame_rgb)

            frame_rgbbb=frame_rgb

            '''
            #自动获取
            hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV)
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 50])
            mask = cv2.inRange(hsv, lower_black, upper_black)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask = cv2.erode(mask, kernel, iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            frame_rgb = frame_rgb[y:y + h, x:x + w]
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_HSV2RGB)
            #cv2.imshow("fff",frame_rgb)
            frame_rgbbb = frame_rgbbb[y:y + h, x:x + w]
            cv2.imshow("fff", frame_rgbbb)
            #cv2.imshow("11111", frame_rgb)
            '''



            # writer.write(frame_rgb)

            # frame_rgb1

            # 获取图像的尺寸
            height, width, channels = frame_rgbbb.shape

            # 计算每个部分的尺寸
            half_height = height // 2
            half_width = width // 2

            # 分割图像
            # 左上
            top_left = frame_rgbbb[0:half_height, 0:half_width]
            # 右上
            top_right = frame_rgbbb[0:half_height, half_width:width]
            # 左下
            bottom_left = frame_rgbbb[half_height:height, 0:half_width]
            # 右下
            bottom_right = frame_rgbbb[half_height:height, half_width:width]

            # 左上
            top_left2 = frame[0:half_height, 0:half_width]
            # 右上
            top_right2 = frame[0:half_height, half_width:width]
            # 左下
            bottom_left2 = frame[half_height:height, 0:half_width]
            # 右下
            bottom_right2 = frame[half_height:height, half_width:width]

            # cv2.imshow("123", top_left)
            '''
            cv2.imshow("123", top_right)
            cv2.imshow("123", bottom_left)
            cv2.imshow("123", bottom_right)

            '''

            # h, w, ch = frame_rgb.shape
            # bytes_per_line = w * ch  # 每个像素3个字节，因为RGB
            # q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # 使用Tesseract进行OCR
            # imageocr = Image.fromarray(frame_rgb)

            # text = pytesseract.image_to_string(imageocr)
            # print(text)

            if frame_id > self.warmup_frame:
                self.pipe_timer.total_time.start()
            if self.heatmap and frame_id % video_fps == 0:#video_fps
                heatmap_count = self.heatmap_detector.run({"image": copy.deepcopy(frame_rgb)})
                heatmap_count[1] = heatmap_count[1].transpose()[::-1]
                plt.imshow(heatmap_count[1])
                # plt.colorbar()
                plt.savefig("temp.jpg")
                # print("heatmap count:", heatmap_count)
                pass
            if self.modebase["idbased"] or self.modebase["skeletonbased"]:
                if frame_id > self.warmup_frame:
                    self.pipe_timer.module_time['mot'].start()

                mot_skip_frame_num = self.mot_predictor.skip_frame_num
                reuse_det_result = False
                if mot_skip_frame_num > 1 and frame_id > 0 and frame_id % mot_skip_frame_num > 0:
                    reuse_det_result = False
                '''
                if frame_id % 40 == 10:
                    res = self.mot_predictor.predict_image(
                        [copy.deepcopy(top_left)],
                        visual=False,
                        reuse_det_result=reuse_det_result)
                    res1=top_left
                    res2=top_left2
                elif frame_id % 40 == 20:

                    res = self.mot_predictor.predict_image(
                        [copy.deepcopy(top_right)],
                        visual=False,
                        reuse_det_result=reuse_det_result)
                    res1=top_right
                    res2=top_right2

                elif frame_id % 40 == 30:

                    res = self.mot_predictor.predict_image(
                        [copy.deepcopy(bottom_left)],
                        visual=False,
                        reuse_det_result=reuse_det_result)
                    res1=bottom_left
                    res2=bottom_left2

                else:

                    res = self.mot_predictor.predict_image(
                        [copy.deepcopy(bottom_right)],
                        visual=False,
                        reuse_det_result=reuse_det_result)
                    res1=bottom_right
                    res2=bottom_right2
                    #print(res)
                    '''


                if frame_id % 10 == 0:
                    res = self.mot_predictor.predict_image(
                        [deepcopy(frame)],
                        visual=False,
                        reuse_det_result=reuse_det_result)
                    res1 = frame_rgbbb
                    res2 = frame

                # mot output format: id, class, score, xmin, ymin, xmax, ymax
                mot_res = parse_mot_res(res)
                # print(mot_res)
                if frame_id > self.warmup_frame:
                    self.pipe_timer.module_time['mot'].end()
                    self.pipe_timer.track_num += len(mot_res['boxes'])

                print("stopstopstopstopstop")

                if frame_id % 10 == 0:
                    print("Thread: {}; trackid number: {}".format(
                        thread_idx, len(mot_res['boxes'])))
                    #cv2.imshow("look",frame)

                    location, jingdu, weidu, classl = getinfo(camera_ids[cameraindex])

                    if (len(mot_res['boxes']) > 1 and cont) or classl==2:  # 人数

                        # 火灾检测
                        blur = cv2.GaussianBlur(frame, (21, 21), 0)
                        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

                        lower1 = [18, 50, 50]
                        upper1 = [35, 255, 255]
                        lower1 = np.array(lower1, dtype="uint8")
                        upper1 = np.array(upper1, dtype="uint8")
                        mask1 = cv2.inRange(hsv, lower1, upper1)

                        output = cv2.bitwise_and(frame, hsv, mask=mask1)
                        no_red = cv2.countNonZero(mask1)

                        # cv2.imshow("output", output)
                        # print("output:", frame)
                        fire = 0
                        if int(no_red) > 20000:
                            fire = 1
                            print('Fire detected')
                        # print(int(no_red))
                        # print("output:".format(mask))
                        fire = 0
                        frame_rgbbb = frame_rgb
                        det = Smoke_File_Detector()
                        print(det.detect_test([frame]))
                        if det.detect_test([frame]) != [[]] and det.detect_test([frame])[0][0]['conf'] > 0.9:
                            fire = 1

                        if fire == 1:
                            h, w, ch = frame_rgb.shape
                            bytes_per_line = w * ch  # 每个像素3个字节，因为RGB
                            q_image = QImage(frame_rgb.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)

                            location, jingdu, weidu, classl = getinfo(camera_ids[cameraindex])
                            iii = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                            # iii="666"
                            print(iii)
                            bostr = str("0")

                            imagepathhh = "C:\\Users\\19025\Desktop\\apache-tomcat-9.0.52\\webapps\\ROOT\\tupian\\"

                            imagepath = imagepathhh + time.strftime("%Y%m%d_%H%M%S", time.localtime()) + ".jpg"

                            q_image.save(imagepath)
                            # q_image.save(imagepathin)

                            files = os.listdir('C:\\Users\\19025\Desktop\\apache-tomcat-9.0.52\\webapps\\ROOT\\tupian')

                            img1str = imagepath
                            iiii=iii+location

                            db_obj1 = address(address=location, shoottime=iiii,
                                              id=random.randint(1, 100000), imagepicpath=imagepath, videooopath=imagepath,
                                              state=bostr, num=10001, recording={"1": ppppnum}, banner="fire",
                                              mourn="fire", jd=jingdu, wd=weidu, cameraindex=camera_ids[cameraindex])

                            db_obj1.save()

                        model6 = Model()

                        label = model6.predict(image=frame)['label']
                        print('predicted label: ', label)
                        confid = model6.predict(image=frame)['confidence']
                        print(confid)

                        if label == "car crash" and confid >= 0.5:
                            h, w, ch = frame_rgb.shape
                            bytes_per_line = w * ch  # 每个像素3个字节，因为RGB
                            q_image = QImage(frame_rgb.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)

                            location, jingdu, weidu, classl = getinfo(camera_ids[cameraindex])
                            iii = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                            # iii="666"
                            print(iii)
                            bostr = str("0")

                            imagepathhh = "C:\\Users\\19025\Desktop\\apache-tomcat-9.0.52\\webapps\\ROOT\\tupian\\"

                            imagepath = imagepathhh + time.strftime("%Y%m%d_%H%M%S", time.localtime()) + ".jpg"

                            q_image.save(imagepath)
                            # q_image.save(imagepathin)

                            files = os.listdir('C:\\Users\\19025\Desktop\\apache-tomcat-9.0.52\\webapps\\ROOT\\tupian')

                            img1str = imagepath
                            iiii = iii + location


                            db_obj1 = address(address=location, shoottime=iiii,
                                              id=random.randint(1, 100000), imagepicpath=imagepath, videooopath=imagepath,
                                              state=bostr, num=10002, recording={"1": ppppnum}, banner="car crash",
                                              mourn="car crash", jd=jingdu, wd=weidu, cameraindex=camera_ids[cameraindex])

                            db_obj1.save()









                        # 报警
                        pppnum = str(len(mot_res['boxes']))
                        ppppnum = int(pppnum)
                        print(mot_res['boxes'][0][0])
                        print(mot_res['boxes'][0][1])
                        print(mot_res['boxes'][0][2])
                        print(mot_res['boxes'][0][3])
                        print(mot_res['boxes'][0][4])
                        print(mot_res['boxes'][0][5])
                        print(mot_res['boxes'][0][6])
                        # print(mot_res['boxes'][0][7])
                        # print(mot_res['boxes'][0][8])
                        print(mot_res['boxes'])
                        print(mot_res['boxes'])
                        print(mot_res['boxes'])
                        print(mot_res['boxes'])
                        print(mot_res['boxes'])
                        print(mot_res['boxes'])
                        print(mot_res['boxes'])
                        print(mot_res['boxes'])

                        frame_rgb = res1

                        mourn = 0

                        h, w, ch = frame_rgb.shape
                        bytes_per_line = w * ch  # 每个像素3个字节，因为RGB
                        q_image = QImage(frame_rgb.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
                        # cv2.imshow("222",frame_rgb)

                        # frame_rgb = res1

                        # 假设 frame_rgb 是您已经加载的RGB格式的numpy.ndarray图像数据
                        frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
                        frame_hsv = cv2.cvtColor(frame_rgb,cv2.COLOR_BGR2HSV)

                        # cv2.imshow("rrrr",frame_rgb)

                        # 将灰度图像转换为二值图像
                        ret, frame_binary = cv2.threshold(frame_gray, 200, 255, cv2.THRESH_BINARY)

                        frame_binaryrr = 255 - frame_binary

                        # cv2.imshow('binary', frame_binaryrr)

                        '''

                        # 打开图片
                        img = frame_rgb
                        # 将图片转换为灰度图，方便处理
                        img_gray = img.convert('L')
                        # 获取图片的尺寸
                        '''

                        cl1 = int(mot_res['boxes'][0][3])
                        cl2 = int(mot_res['boxes'][0][4])
                        cl3 = int(mot_res['boxes'][0][5])
                        cl4 = int(mot_res['boxes'][0][6])

                        #onecloth=frame_hsv[mot_res['boxes'][0][5]:mot_res['boxes'][0][6]][mot_res['boxes'][0][3]:mot_res['boxes'][0][4]]

                        print(mot_res['boxes'])
                        cl2, cl3 = cl3, cl2

                        if True or abs(cl1 - cl2) > 2 and abs(cl3 - cl4) > 2:
                            '''

                            if cl1 >cl2:
                                cl1,cl2=cl2,cl1
                            if cl3>cl4:
                                cl3,cl4=cl4,cl3
                                '''

                            widthg2 = abs(mot_res['boxes'][0][4] - mot_res['boxes'][0][3])
                            heightg2 = abs(mot_res['boxes'][0][6] - mot_res['boxes'][0][5])

                            frame_graycloth = frame_gray[cl3:cl4][cl1:cl2]
                            # frame_graycloth = cv2.cvtColor(frame_rgbcloth, cv2.COLOR_BGR2GRAY)
                            widthg, heightg = abs(cl2 - cl1), abs(cl4 - cl3)

                            # 初始化白色像素计数器
                            white_count = 0
                            # cv2.imshow("cl",frame_graycloth)
                            print(frame_gray)

                            # 遍历图片的每个像素
                            for x in range(cl3, cl4):
                                for y in range(cl1, cl2):
                                    # 如果像素值大于240（接近白色），则认为是白色
                                    try:
                                        if frame_gray[x][y] > 240:
                                            white_count += 1
                                    except:
                                        continue

                            # 计算白色像素的比例
                            total_pixels = widthg2 * heightg2
                            white_ratio = float(white_count) / float(total_pixels)

                            # 判断白色像素是否超过70%
                            if white_ratio > 0.75:
                                print(white_ratio)
                                print(88888888888888888888888888888888888888)
                                print(total_pixels)
                                print(white_count)
                                mourn = 1
                            else:
                                print(99999999999999999999999999999999999999)

                        # 使用Tesseract进行OCR
                        '''imageocr = Image.fromarray(frame_binaryrr)
                        imageocr1 = Image.fromarray(frame_rgb)
                        '''

                        # 初始化 PaddleOCR，设置中文识别
                        ocr = PaddleOCR(use_angle_cls=True, lang="ch")

                        # 使用 PaddleOCR 进行 OCR
                        result = ocr.ocr(frame_gray, cls=True)

                        print(result)

                        # 读取图像
                        image = frame  # 替换为你的图片路径

                        # 转换为HSV色彩空间
                        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                        # 定义红色的阈值范围
                        lower_red1 = np.array([0, 100, 100])
                        upper_red1 = np.array([10, 255, 255])
                        lower_red2 = np.array([160, 100, 100])
                        upper_red2 = np.array([180, 255, 255])

                        # 创建红色掩膜
                        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                        mask = cv2.bitwise_or(mask1, mask2)

                        # 进行形态学操作，去噪声
                        kernel = np.ones((5, 5), np.uint8)
                        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                        # 查找轮廓
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        rcttext = [None]
                        print(cameraindex)
                        print(cameraindex)
                        print(cameraindex)

                        # 提取文字区域
                        for contour in contours:
                            x, y, w, h = cv2.boundingRect(contour)

                            # 设置一个条件来过滤掉小的区域

                            if w > 7 and h > 1:  # 可以根据需要调整
                                # 提取横幅区域
                                banner_region = image[y:y + h, x:x + w]

                                # 使用Pytesseract进行文字识别
                                ocr = PaddleOCR(use_angle_cls=True, lang="ch")
                                rcttext = ocr.ocr(banner_region, cls=True)
                                # text = pytesseract.image_to_string(banner_region, lang='chi_sim')  # 根据需要选择语言
                                if rcttext != [None]:
                                    print("识别到的文字:", rcttext)

                                    # 在原图上绘制矩形框
                                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                                    # 显示结果图像
                                    cv2.imshow('Detected Banners', image)
                                    # cv2.waitKey(0)
                                    # cv2.destroyAllWindows()

                        text7 = ""
                        if rcttext != [None]:
                            print("rcttext is not none")
                            print("888")
                            print(rcttext)
                            print("888")
                            for line in rcttext:
                                # line[1] 是识别到的文本字符串
                                try:
                                    # 直接对字符串调用 strip()
                                    for iii in line:
                                        text7 += str(iii[1][0]).strip() + " "

                                # 捕获可能出现的错误，例如当识别结果不是预期的格式时
                                except IndexError:
                                    print("Error: OCR result format is not as expected.")
                            text7 = text7.strip()
                        else:
                            print("rectext is none")
                            text7 = "有横幅未提取到文字"

                        # 使用列表推导式提取每个识别结果中的文本内容
                        text5 = ""

                        # 遍历识别结果，提取文本内容，并去除每行文本的首尾空白字符
                        if result != [None]:
                            for line in result:
                                # line[1] 是识别到的文本字符串
                                try:
                                    # 直接对字符串调用 strip()
                                    for iii in line:
                                        text5 += str(iii[1][0]).strip() + " "

                                # 捕获可能出现的错误，例如当识别结果不是预期的格式时
                                except IndexError:
                                    print("Error: OCR result format is not as expected.")
                            text5 = text5.strip()
                        else:
                            text5 = ""

                        # 去掉字符串最后的换行符
                        # text5 = text5.strip()
                        # 打印识别结果
                        #      for line in result:
                        #         # line 是一个列表，包含 [position, text, confidence]
                        #        print(f"Text: {line[1]}, Confidence: {line[2]}")
                        # 如果你想去除文本中的空格和换行符，可以像下面这样做
                        #      text4 = ''.join([line[1] for line in result]).strip()
                        #     print(text4)
                        # text1 = str(pytesseract.image_to_string(imageocr, lang='chi_sim'))
                        # print(text1)
                        # text2=""
                        # for i in text1:
                        # if i != " " or i != "\n":
                        # text2 += i
                        # text3 = text1.replace(" ", "").replace("\n", "").replace("\r", "")

                        # audio_file_path = 'C:\\Users\\ls\\Desktop\\bj.mp3'  # 替换为你的音频文件路径
                        # os.system(f'start {audio_file_path}')

                        writer.release()
                        out1 = os.path.join("C:\\Users\\19025\Desktop\\apache-tomcat-9.0.52\\webapps\\ROOT\\shipin",
                                            time.strftime("%Y%m%d_%H%M%S", time.localtime()) + ".mp4")
                        out2 = "C:\\Users\\19025\Desktop\\apache-tomcat-9.0.52\\webapps\\ROOT\\tupian\\" + str(out1)
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        # fourcc=-1
                        writer = cv2.VideoWriter(out1, fourcc, fps - 45, (width, height))
                        print('save result to {}'.format(out1))

                        imagepathhh = "C:\\Users\\19025\Desktop\\apache-tomcat-9.0.52\\webapps\\ROOT\\tupian\\"

                        imagepath = imagepathhh + time.strftime("%Y%m%d_%H%M%S", time.localtime()) + ".jpg"

                        # imagepathin = "C:\\Users\\19025\Desktop\\crowd_vis-main\\pp-human\\pipeline\\tutupian\\" + time.strftime("%Y%m%d_%H%M%S", time.localtime()) + ".jpg"
                        # imagepath = "C:\\Users\\ls\\Desktop\\humancut\\"
                        q_image.save(imagepath)
                        # q_image.save(imagepathin)

                        files = os.listdir('C:\\Users\\19025\Desktop\\apache-tomcat-9.0.52\\webapps\\ROOT\\tupian')

                        img1str = imagepath

                        alld = 0






                        # file_path = os.path.join('C:\\Users\\19025\Desktop\\crowd_vis-main\\pp-human\\pipeline\\tutupian', file)
                        #file_path1 = os.path.join('C:\\Users\\19025\Desktop\\apache-tomcat-9.0.52\\webapps\\ROOT\\tupian', file)

                        #print(file_path1)
              #9 11 0 7

                        t = str(time.strftime("%Y%m%d_%H%M%S", time.localtime()))
                        print(t[9:11])
                        print(t)

                        lastr=0


                        try:
                            if setlocal==0:
                                addresses = address.objects.filter(cameraindex=camera_ids[cameraindex])
                            else:
                                addresses = address.objects.filter(cameraindex=0)


                            instance = max(addresses, key=lambda x: x.shoottime)
                            lastr=1
                        except:
                            print("exc")
                            lastr=0








                        if lastr:

                            lasttime = max(instance.recording.keys())
                            if len(lasttime) ==1:
                                lasttime=instance.shoottime
                            print(t)
                            print(lasttime)
                            print(t)
                            print(lasttime)
                            print(t)
                            print(lasttime)
                            print(t)
                            print(lasttime)
                            print(t)

                            print(lasttime)
                            print(t)
                            print(lasttime)
                            print(lasttime)
                            print(lasttime)
                            print(lasttime)
                            print(lasttime)
                            print(lasttime)
                            print(lasttime)
                            print(t)
                            print(t)
                            print(t)
                            print(t)
                            print(t)
                            print(t)
                            print(lasttime[11:13])
                            print(t[9:11])
                            print(lasttime[11:13])
                            print(t[9:11])
                            print(lasttime[11:13])
                            print(t[9:11])
                            print(lasttime[11:13])
                            print(t[9:11])
                            print(lasttime[11:13])
                            print(t[9:11])
                            print(lasttime[11:13])
                            print(t[9:11])
                            print(lasttime[11:13])
                            print(t[9:11])

                            print(instance.recording.values())
                            print(instance.recording.values())
                            print(instance.recording.values())

                            print(instance.recording.values())
                            print(instance.recording.values())
                            print(instance.recording.values())

                            instancelist=list(instance.recording.values())
                            '''for i in instancelist:
                                i=str(i)'''

                            max_people = max(instance.recording.values())
                            instance.num = max_people
                            now = datetime.now()
                            nowtime = now.strftime('%Y-%m-%d %H:%M:%S')

                            instance.recording[nowtime] = len(mot_res['boxes'])
                            update_json = json.dumps(instance.recording)

                            instance.json_data = update_json

                            instance.save()
                            alld = 1

                            daycheck = 0
                            if (t[0:4] == lasttime[0:4]) and (t[4:6] == lasttime[5:7]) and (t[6:8] == lasttime[8:10]):
                                daycheck = 1
                            if not daycheck:
                                alld = 0
                                iii = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                                # iii="666"
                                print(iii)
                                bostr = str("0")
                                location, jingdu, weidu, classl = getinfo(camera_ids[cameraindex])
                                if (classl != 1) and (classl != 2):
                                    iiii = iii + location
                                    db_obj1 = address(address=location, shoottime=iiii,
                                                      id=random.randint(1, 100000), imagepicpath=imagepath,
                                                      videooopath=out2,
                                                      state=bostr, num=pppnum, recording={"1": ppppnum}, banner=text7,
                                                      mourn=mourn, jd=jingdu, wd=weidu,
                                                      cameraindex=camera_ids[cameraindex])
                                    db_obj1.save()
                                if (classl ==2 or classl ==0):

                                    for idx, i in enumerate(len(mot_res['boxes'])):
                                        onecloth = frame_hsv[mot_res['boxes'][i][5]:mot_res['boxes'][i][6]][mot_res['boxes'][i][3]:mot_res['boxes'][i][4]]
                                        # 存储每件衣服的直方图
                                        clothing_histograms = []
                                        # 存储每件衣服的相似性分组
                                        clothing_groups = defaultdict(list)



                                        # 提取整张图片的颜色直方图
                                        hist = extract_clothing_histogram(image, color_space='hsv')

                                        # 将直方图存储到列表中
                                        clothing_histograms.append(hist)

                                        # 比较当前衣服与其他衣服的相似性
                                        for i in range(len(clothing_histograms) - 1):
                                            similarity = compare_histograms(clothing_histograms[-1],
                                                                            clothing_histograms[i])
                                            if similarity > 0.6:  # 相似性阈值
                                                clothing_groups[idx].append(i)

                                        # 统计是否有3个及以上的人穿同一件衣服
                                        for group_id, group in clothing_groups.items():
                                            if len(group) >= 4:
                                                print(f"发现4个及以上的人穿同一件衣服，组ID: {group_id}, 成员: {group}")
                                                iiii = iii + location
                                                db_obj1 = address(address=location, shoottime=iiii,
                                                                  id=random.randint(1, 100000), imagepicpath=imagepath,
                                                                  videooopath=out2,
                                                                  state=bostr, num=pppnum, recording={"1": ppppnum},
                                                                  banner=text7,
                                                                  mourn=mourn, jd=jingdu, wd=weidu,
                                                                  cameraindex=camera_ids[cameraindex])
                                                db_obj1.save()



                                    '''
                                    if text5=="":
                                        numo1=0
                                        numo2=0
                                        db_obj1 = address(address=text5, shoottime=iii,
                                                          id=random.randint(1, 1000), imagepicpath=imagepath, videooopath=out2,
                                                          state=bostr, num=pppnum, recording={"1": "1"}, banner=text7,
                                                          mourn=mourn, jd=numo1, wd=numo2)
                                    else:
                                        numo1, numo2,textgover = find_numbers(text5, data)
                                        db_obj1 = address(address=textgover, shoottime=iii,
                                                          id=random.randint(1, 1000), imagepicpath=imagepath, videooopath=out2,
                                                          state=bostr, num=pppnum, recording={"1": "1"}, banner=text7,
                                                          mourn=mourn, jd=numo1, wd=numo2)
                                    '''





                                if classl==1:



                                    schoolalarm=0
                                    if text7!="" or mourn:
                                        schoolalarm=1




                                    if schoolalarm:
                                        iiii = iii + location
                                        db_obj1 = address(address=location, shoottime=iiii,
                                                          id=random.randint(1, 100000), imagepicpath=imagepath,
                                                          videooopath=out2,
                                                          state=bostr, num=pppnum, recording={"1": ppppnum}, banner=text7,
                                                          mourn=mourn, jd=jingdu, wd=weidu,
                                                          cameraindex=camera_ids[cameraindex])
                                        db_obj1.save()




                            if  int(t[9:11])-int(lasttime[11:13])>2 and daycheck:
                                alld = 0
                                iii = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                                # iii="666"
                                print(iii)
                                bostr = str("0")
                                location, jingdu, weidu, classl = getinfo(camera_ids[cameraindex])
                                if (classl != 1) and (classl != 2):
                                    iiii = iii + location

                                    db_obj1 = address(address=location, shoottime=iiii,
                                                      id=random.randint(1, 100000), imagepicpath=imagepath,
                                                      videooopath=out2,
                                                      state=bostr, num=pppnum, recording={"1": ppppnum}, banner=text7,
                                                      mourn=mourn, jd=jingdu, wd=weidu,
                                                      cameraindex=camera_ids[cameraindex])
                                    '''
                                    if text5=="":
                                        numo1=0
                                        numo2=0
                                        db_obj1 = address(address=text5, shoottime=iii,
                                                          id=random.randint(1, 1000), imagepicpath=imagepath, videooopath=out2,
                                                          state=bostr, num=pppnum, recording={"1": "1"}, banner=text7,
                                                          mourn=mourn, jd=numo1, wd=numo2)
                                    else:
                                        numo1, numo2,textgover = find_numbers(text5, data)
                                        db_obj1 = address(address=textgover, shoottime=iii,
                                                          id=random.randint(1, 1000), imagepicpath=imagepath, videooopath=out2,
                                                          state=bostr, num=pppnum, recording={"1": "1"}, banner=text7,
                                                          mourn=mourn, jd=numo1, wd=numo2)
                                    '''

                                    db_obj1.save()



                            if (int(t[9:11])-int(lasttime[11:13])<=2) and daycheck:

                                '''
                                print(file_path1)
                                instance = address.objects.get(imagepicpath=file_path1)
                                '''



                                # json_data = json.loads(instance.recording)
                                max_people = max(instance.recording.values(), default=0)
                                instance.num = max_people
                                now = datetime.now()
                                nowtime = now.strftime('%Y-%m-%d %H:%M:%S')

                                instance.recording[nowtime] = len(mot_res['boxes'])
                                update_json = json.dumps(instance.recording)

                                instance.json_data = update_json

                                instance.save()
                                alld = 1

                                # if sim >= 0.4:
                                # break

                        # app = QApplication(sys.argv)
                        humannum = str(len(mot_res['boxes']))
                        # window = ImageWindow(q_image,text5,humannum)
                        # gettextinfo(text5)

                        # window.show()
                        # time.sleep(2000)

                        # app.exec_()

                        # sys.exit()

                        if not lastr:
                            iii = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                            # iii="666"
                            print(iii)
                            bostr = str("0")
                            location, jingdu, weidu, classl = getinfo(camera_ids[cameraindex])

                            if (classl != 1) and (classl != 2):
                                iiii = iii + location

                                db_obj1 = address(address=location, shoottime=iiii,
                                                  id=random.randint(1, 100000), imagepicpath=imagepath, videooopath=out2,
                                                  state=bostr, num=pppnum, recording={"1": ppppnum}, banner=text7,
                                                  mourn=mourn, jd=jingdu, wd=weidu, cameraindex=camera_ids[cameraindex])
                                '''
                                if text5=="":
                                    numo1=0
                                    numo2=0
                                    db_obj1 = address(address=text5, shoottime=iii,
                                                      id=random.randint(1, 1000), imagepicpath=imagepath, videooopath=out2,
                                                      state=bostr, num=pppnum, recording={"1": "1"}, banner=text7,
                                                      mourn=mourn, jd=numo1, wd=numo2)
                                else:
                                    numo1, numo2,textgover = find_numbers(text5, data)
                                    db_obj1 = address(address=textgover, shoottime=iii,
                                                      id=random.randint(1, 1000), imagepicpath=imagepath, videooopath=out2,
                                                      state=bostr, num=pppnum, recording={"1": "1"}, banner=text7,
                                                      mourn=mourn, jd=numo1, wd=numo2)
                                '''

                                db_obj1.save()

                        start_time = time.time()

                        # cont = False

                    '''

                    if not cont:
                        if (time.time() - start_time) > 10:#录视频多少秒


                            iii=str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                            #iii="666"
                            print(iii)
                            bostr=str("0")


                            db_obj1 = address(address=text5, shoottime=iii,
                                              id=random.randint(1, 1000),imagepicpath=imagepath,videooopath=out2,state= bostr,num=pppnum,recording={"1":"111"})
                            db_obj1.save()




                            instance = address.objects.get(id=747)



                            #json_data = json.loads(instance.recording)
                            instance.recording['1'] = iii
                            update_json = json.dumps(instance.recording)


                            instance.json_data = update_json

                            instance.save()


                            writer.release()
                            cont = True

                    '''

                # flow_statistic only support single class MOT
                boxes, scores, ids = res[0]  # batch size = 1 in MOT
                mot_result = (frame_id + 1, boxes[0], scores[0],
                              ids[0])  # single class
                statistic = flow_statistic(
                    mot_result,
                    self.secs_interval,
                    self.do_entrance_counting,
                    self.do_break_in_counting,
                    self.region_type,
                    video_fps,
                    entrance,
                    id_set,
                    interval_id_set,
                    in_id_list,
                    out_id_list,
                    prev_center,
                    records,
                    ids2names=self.mot_predictor.pred_config.labels)
                records = statistic['records']

                if self.illegal_parking_time != -1:
                    object_in_region_info, illegal_parking_dict = update_object_info(
                        object_in_region_info, mot_result, self.region_type,
                        entrance, video_fps, self.illegal_parking_time)
                    if len(illegal_parking_dict) != 0:
                        # build relationship between id and plate
                        for key, value in illegal_parking_dict.items():
                            plate = self.collector.get_carlp(key)
                            illegal_parking_dict[key]['plate'] = plate

                # nothing detected
                if len(mot_res['boxes']) == 0:

                    frame_id += 1
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.img_num += 1
                        self.pipe_timer.total_time.end()
                    if self.cfg['visual']:
                        _, _, fps = self.pipe_timer.get_total_time()
                        im = self.visualize_video(frame, mot_res, frame_id, fps,
                                                  entrance, records,
                                                  center_traj)  # visualize
                        if len(self.pushurl) > 0:

                            pushstream.pipe.stdin.write(im.tobytes())
                        else:

                            writer.write(im)
                            if self.file_name is None:
                                # use camera_id
                                cv2.imshow('Paddle-Pipeline', im)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break
                    continue

                self.pipeline_res.update(mot_res, 'mot')
                crop_input, new_bboxes, ori_bboxes = crop_image_with_mot(
                    frame_rgb, mot_res)

                if self.with_vehicleplate and frame_id % 10 == 0:
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['vehicleplate'].start()
                    plate_input, _, _ = crop_image_with_mot(
                        frame_rgb, mot_res, expand=False)
                    platelicense = self.vehicleplate_detector.get_platelicense(
                        plate_input)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['vehicleplate'].end()
                    self.pipeline_res.update(platelicense, 'vehicleplate')
                else:
                    self.pipeline_res.clear('vehicleplate')

                if self.with_human_attr:
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['attr'].start()
                    attr_res = self.attr_predictor.predict_image(
                        crop_input, visual=False)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['attr'].end()
                    self.pipeline_res.update(attr_res, 'attr')

                if self.with_vehicle_attr:
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['vehicle_attr'].start()
                    attr_res = self.vehicle_attr_predictor.predict_image(
                        crop_input, visual=False)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['vehicle_attr'].end()
                    self.pipeline_res.update(attr_res, 'vehicle_attr')

                if self.with_idbased_detaction:
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['det_action'].start()
                    det_action_res = self.det_action_predictor.predict(
                        crop_input, mot_res)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['det_action'].end()
                    self.pipeline_res.update(det_action_res, 'det_action')

                    if self.cfg['visual']:
                        self.det_action_visual_helper.update(det_action_res)

                if self.with_idbased_clsaction:
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['cls_action'].start()
                    cls_action_res = self.cls_action_predictor.predict_with_mot(
                        crop_input, mot_res)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['cls_action'].end()
                    self.pipeline_res.update(cls_action_res, 'cls_action')

                    if self.cfg['visual']:
                        self.cls_action_visual_helper.update(cls_action_res)

                if self.with_skeleton_action:
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['kpt'].start()
                    kpt_pred = self.kpt_predictor.predict_image(
                        crop_input, visual=False)
                    keypoint_vector, score_vector = translate_to_ori_images(
                        kpt_pred, np.array(new_bboxes))
                    kpt_res = {}
                    kpt_res['keypoint'] = [
                        keypoint_vector.tolist(), score_vector.tolist()
                    ] if len(keypoint_vector) > 0 else [[], []]
                    kpt_res['bbox'] = ori_bboxes
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['kpt'].end()

                    self.pipeline_res.update(kpt_res, 'kpt')

                    self.kpt_buff.update(kpt_res, mot_res)  # collect kpt output
                    state = self.kpt_buff.get_state(
                    )  # whether frame num is enough or lost tracker

                    skeleton_action_res = {}
                    if state:
                        if frame_id > self.warmup_frame:
                            self.pipe_timer.module_time[
                                'skeleton_action'].start()
                        collected_keypoint = self.kpt_buff.get_collected_keypoint(
                        )  # reoragnize kpt output with ID
                        skeleton_action_input = parse_mot_keypoint(
                            collected_keypoint, self.coord_size)
                        skeleton_action_res = self.skeleton_action_predictor.predict_skeleton_with_mot(
                            skeleton_action_input)
                        if frame_id > self.warmup_frame:
                            self.pipe_timer.module_time['skeleton_action'].end()
                        self.pipeline_res.update(skeleton_action_res,
                                                 'skeleton_action')

                    if self.cfg['visual']:
                        self.skeleton_action_visual_helper.update(
                            skeleton_action_res)

                if self.with_mtmct and frame_id % 10 == 0:
                    crop_input, img_qualities, rects = self.reid_predictor.crop_image_with_mot(
                        frame_rgb, mot_res)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['reid'].start()
                    reid_res = self.reid_predictor.predict_batch(crop_input)

                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['reid'].end()

                    reid_res_dict = {
                        'features': reid_res,
                        "qualities": img_qualities,
                        "rects": rects
                    }
                    self.pipeline_res.update(reid_res_dict, 'reid')
                else:
                    self.pipeline_res.clear('reid')

            if self.with_video_action:
                # get the params
                frame_len = self.cfg["VIDEO_ACTION"]["frame_len"]
                sample_freq = self.cfg["VIDEO_ACTION"]["sample_freq"]

                if sample_freq * frame_len > frame_count:  # video is too short
                    sample_freq = int(frame_count / frame_len)

                # filter the warmup frames
                if frame_id > self.warmup_frame:
                    self.pipe_timer.module_time['video_action'].start()

                # collect frames
                if frame_id % 20 == 0:
                    # Scale image
                    scaled_img = scale(frame_rgb)
                    video_action_imgs.append(scaled_img)

                # the number of collected frames is enough to predict video action
                if len(video_action_imgs) == frame_len:
                    # print(video_action_imgs)
                    # video_action_imgs = video_action_imgs.tolist()

                    for vaimg in video_action_imgs:
                        # print(vaimg)
                        vaimg = np.array(vaimg)
                        vaimg = vaimg.tolist()

                        # print(vaimg)
                        vaimg = tuple(vaimg)
                        # print(vaimg)
                    # print(video_action_imgs)

                    classes, scores = self.video_action_predictor.predict(
                        video_action_imgs)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['video_action'].end()

                    video_action_res = {"class": classes[0], "score": scores[0]}
                    print(classes)
                    self.pipeline_res.update(video_action_res, 'video_action')

                    print("video_action_res:", video_action_res)
                    try:
                        instance = address.objects.get(shoottime=iii)
                    except:
                        print("exc")
                        continue

                    # json_data = json.loads(instance.recording)
                    '''now = datetime.now()
                    nowtime = now.strftime('%Y-%m-%d %H:%M:%S')'''

                    instance.fight = classes[0]
                    '''update_json = json.dumps(instance.recording)

                    instance.json_data = update_json'''

                    instance.save()

                    video_action_imgs.clear()  # next clip

            self.collector.append(frame_id, self.pipeline_res)

            if frame_id > self.warmup_frame:
                self.pipe_timer.img_num += 1
                self.pipe_timer.total_time.end()
            frame_id += 1

            if self.cfg['visual']:
                _, _, fps = self.pipe_timer.get_total_time()

                im = self.visualize_video(frame, self.pipeline_res,
                                          self.collector, frame_id, fps,
                                          entrance, records, center_traj,
                                          self.illegal_parking_time != -1,
                                          illegal_parking_dict)  # visualize
                if self.heatmap:
                    heat_img = cv2.imread("temp.jpg")
                    img2 = cv2.resize(heat_img, (int(heat_img.shape[0] / 1.5), int(heat_img.shape[1] / 1.5)))

                    # I want to put logo on top-left corner, So I create a ROI
                    # 首先获取原始图像roi
                    rows, cols, channels = img2.shape
                    roi = im[0:rows, 0:cols]

                    # 原始图像转化为灰度值
                    # Now create a mask of logo and create its inverse mask also
                    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                    '''
                    将一个灰色的图片，变成要么是白色要么就是黑色。（大于规定thresh值就是设置的最大值（常为255，也就是白色））
                    '''
                    # 将灰度值二值化，得到ROI区域掩模
                    ret, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY)
                    # ROI掩模区域反向掩模
                    mask_inv = cv2.bitwise_not(mask)
                    # 掩模显示背景
                    # Now black-out the area of logo in ROI
                    img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
                    # 掩模显示前景
                    # Take only region of logo from logo image.
                    img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)
                    # 前背景图像叠加
                    # Put logo in ROI and modify the main image
                    dst = cv2.add(img1_bg, img2_fg)
                    im[0:rows, 0:cols] = dst
                    im[0:rows, 0:cols] = dst

                    # cv2.imshow('res', im)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                show_im = cv2.resize(im, (0, 0), fx=0.5, fy=0.5)
                if return_im:
                    if self.heatmap and frame_id % video_fps == 0:
                        yield {"im": show_im, "heatcount": heatmap_count[0], "heatmap": heatmap_count[1]}
                    elif frame_id % video_fps == 0:
                        yield {"im": show_im, "mot_res": [records[-1], len(mot_res['boxes'])]}
                    else:
                        yield {"im": show_im}
                # temp1, temp2 = cv2.imencode('.jpeg', show_im)
                # fp = open('data/frame-' + str(datetime.datetime.now().timestamp()), 'wb')
                # fp.write(temp2.tobytes())
                # fp.close()
                # break
                # fp = open('data/records' + '.txt', 'w')
                # ajax_data = records[-1]
                # ajax_data = [ajax_data, len(mot_res['boxes']), 0]
                # json.dump(ajax_data, fp)
                # fp.close()
                if len(self.pushurl) > 0:
                    pushstream.pipe.stdin.write(im.tobytes())
                else:

                    writer.write(im)
                    if self.file_name is None or True:
                        # use camera_id

                        # cv2.imshow('Paddle-Pipeline', im)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

        # self.cfg['visual'] = True
        # print(321)

        if self.cfg['visual'] and len(self.pushurl) == 0:
            print("w3")
            writer.release()
            print('save result to {}'.format(out_path))
        # sys.exit()
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    def visualize_video(self,
                        image,
                        result,
                        collector,
                        frame_id,
                        fps,
                        entrance=None,
                        records=None,
                        center_traj=None,
                        do_illegal_parking_recognition=False,
                        illegal_parking_dict=None):
        mot_res = deepcopy(result.get('mot'))
        if mot_res is not None:
            ids = mot_res['boxes'][:, 0]
            scores = mot_res['boxes'][:, 2]
            boxes = mot_res['boxes'][:, 3:]
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        else:
            boxes = np.zeros([0, 4])
            ids = np.zeros([0])
            scores = np.zeros([0])

        # single class, still need to be defaultdict type for ploting
        num_classes = 1
        online_tlwhs = defaultdict(list)
        online_scores = defaultdict(list)
        online_ids = defaultdict(list)
        online_tlwhs[0] = boxes
        online_scores[0] = scores
        online_ids[0] = ids

        if mot_res is not None:
            image = plot_tracking_dict(
                image,
                num_classes,
                online_tlwhs,
                online_ids,
                online_scores,
                frame_id=frame_id,
                fps=fps,
                ids2names=self.mot_predictor.pred_config.labels,
                do_entrance_counting=self.do_entrance_counting,
                do_break_in_counting=self.do_break_in_counting,
                do_illegal_parking_recognition=do_illegal_parking_recognition,
                illegal_parking_dict=illegal_parking_dict,
                entrance=entrance,
                records=records,
                center_traj=center_traj)

        human_attr_res = result.get('attr')
        if human_attr_res is not None:
            boxes = mot_res['boxes'][:, 1:]
            human_attr_res = human_attr_res['output']
            image = visualize_attr(image, human_attr_res, boxes)
            image = np.array(image)

        vehicle_attr_res = result.get('vehicle_attr')
        if vehicle_attr_res is not None:
            boxes = mot_res['boxes'][:, 1:]
            vehicle_attr_res = vehicle_attr_res['output']
            image = visualize_attr(image, vehicle_attr_res, boxes)
            image = np.array(image)

        if mot_res is not None:
            vehicleplate = False
            plates = []
            for trackid in mot_res['boxes'][:, 0]:
                plate = collector.get_carlp(trackid)
                if plate != None:
                    vehicleplate = True
                    plates.append(plate)
                else:
                    plates.append("")
            if vehicleplate:
                boxes = mot_res['boxes'][:, 1:]
                image = visualize_vehicleplate(image, plates, boxes)
                image = np.array(image)

        kpt_res = result.get('kpt')
        if kpt_res is not None:
            image = visualize_pose(
                image,
                kpt_res,
                visual_thresh=self.cfg['kpt_thresh'],
                returnimg=True)

        video_action_res = result.get('video_action')
        if video_action_res is not None:
            video_action_score = None
            if video_action_res and video_action_res["class"] == 1:
                video_action_score = video_action_res["score"]
            mot_boxes = None
            if mot_res:
                mot_boxes = mot_res['boxes']
            image = visualize_action(
                image,
                mot_boxes,
                action_visual_collector=None,
                action_text="SkeletonAction",
                video_action_score=video_action_score,
                video_action_text="Fight")

        visual_helper_for_display = []
        action_to_display = []

        skeleton_action_res = result.get('skeleton_action')
        if skeleton_action_res is not None:
            visual_helper_for_display.append(self.skeleton_action_visual_helper)
            action_to_display.append("Falling")

        det_action_res = result.get('det_action')
        if det_action_res is not None:
            visual_helper_for_display.append(self.det_action_visual_helper)
            action_to_display.append("Smoking")

        cls_action_res = result.get('cls_action')
        if cls_action_res is not None:
            visual_helper_for_display.append(self.cls_action_visual_helper)
            action_to_display.append("Calling")

        if len(visual_helper_for_display) > 0:
            image = visualize_action(image, mot_res['boxes'],
                                     visual_helper_for_display,
                                     action_to_display)

        return image

    def visualize_image(self, im_files, images, result):
        start_idx, boxes_num_i = 0, 0
        det_res = result.get('det')
        human_attr_res = result.get('attr')
        vehicle_attr_res = result.get('vehicle_attr')
        vehicleplate_res = result.get('vehicleplate')

        for i, (im_file, im) in enumerate(zip(im_files, images)):
            if det_res is not None:
                det_res_i = {}
                boxes_num_i = det_res['boxes_num'][i]
                det_res_i['boxes'] = det_res['boxes'][start_idx:start_idx +
                                                                boxes_num_i, :]
                im = visualize_box_mask(
                    im,
                    det_res_i,
                    labels=['target'],
                    threshold=self.cfg['crop_thresh'])
                im = np.ascontiguousarray(np.copy(im))
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            if human_attr_res is not None:
                human_attr_res_i = human_attr_res['output'][start_idx:start_idx
                                                                      + boxes_num_i]
                im = visualize_attr(im, human_attr_res_i, det_res_i['boxes'])
            if vehicle_attr_res is not None:
                vehicle_attr_res_i = vehicle_attr_res['output'][
                                     start_idx:start_idx + boxes_num_i]
                im = visualize_attr(im, vehicle_attr_res_i, det_res_i['boxes'])
            if vehicleplate_res is not None:
                plates = vehicleplate_res['vehicleplate']
                det_res_i['boxes'][:, 4:6] = det_res_i[
                                                 'boxes'][:, 4:6] - det_res_i['boxes'][:, 2:4]
                im = visualize_vehicleplate(im, plates, det_res_i['boxes'])

            img_name = os.path.split(im_file)[-1]
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            out_path = os.path.join(self.output_dir, img_name)
            cv2.imwrite(out_path, im)
            print("save result to: " + out_path)
            start_idx += boxes_num_i


def main():
    cfg = merge_cfg(FLAGS)  # use command params to update config
    print_arguments(cfg)

    pipeline = Pipeline(FLAGS, cfg)
    # pipeline.run()
    pipeline.run_multithreads()


def main_new(FLAGS, video_file):
    cfg = merge_cfg(FLAGS)  # use command params to update config
    print_arguments(cfg)

    pipeline = PipePredictor(FLAGS, cfg)
    pipeline.set_file_name(video_file)
    res = pipeline.predict_video(video_file, return_im=True)
    for i in res:
        yield i


if __name__ == '__main__':
    paddle.enable_static()

    # parse params from command
    parser = argsparser()
    FLAGS = parser.parse_args()
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU'
                            ], "device should be CPU, GPU or XPU"

    main()

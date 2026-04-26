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
from types import SimpleNamespace
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt
from PIL import Image
from paddleocr import PaddleOCR, draw_ocr
from collections import defaultdict, deque
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)
sys.path.insert(0, os.path.dirname(__file__))

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
import argparse
import closimi3
import torch.backends.cudnn as cudnn
from models.experimental import *
from utils1.datasets import *
from utils1.general import *
from utils1 import torch_utils
import gc

setlocal = 1


class VirtualTwinAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=4):
        super(VirtualTwinAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.W = nn.Linear(in_features, out_features * num_heads, bias=False)
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * out_features))
        self.U = nn.Linear(in_features, out_features * num_heads, bias=False)
        self.V = nn.Linear(out_features * num_heads, out_features * num_heads, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
        nn.init.xavier_uniform_(self.U.weight)
        nn.init.xavier_uniform_(self.V.weight)

    def forward(self, h, adj, edge_weights):
        N = h.size(0)
        h_trans = self.W(h).view(N, self.num_heads, self.out_features)

        a_input = torch.cat([h_trans.repeat(1, 1, N).view(N * N, self.num_heads, -1),
                             h_trans.repeat(N, 1, 1).view(N * N, self.num_heads, -1)], dim=2)

        e_raw = (a_input * self.a.view(1, self.num_heads, -1)).sum(dim=2).view(N, N, self.num_heads)
        e_raw = self.leaky_relu(e_raw)

        if edge_weights is not None:
            edge_weights = edge_weights.unsqueeze(-1).expand_as(e_raw)
            e_modulated = e_raw * edge_weights
        else:
            e_modulated = e_raw

        zero_vec = -9e15 * torch.ones_like(e_modulated)
        attention = torch.where(adj.unsqueeze(-1) > 0, e_modulated, zero_vec)
        alpha = F.softmax(attention, dim=1)

        c_virtual = torch.zeros(N, self.num_heads, self.out_features).to(h.device)
        for k in range(self.num_heads):
            c_virtual[:, k, :] = torch.matmul(alpha[:, :, k], h_trans[:, k, :])

        c_virtual = c_virtual.view(N, -1)

        h_next = torch.sigmoid(self.U(h) + self.V(c_virtual))
        return h_next


class VTAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(VTAN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(VirtualTwinAttentionLayer(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(VirtualTwinAttentionLayer(hidden_dim * 4, hidden_dim))
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 4, output_dim),
            nn.Tanh()
        )

    def forward(self, x, adj, edge_weights):
        h = x
        for layer in self.layers:
            h = layer(h, adj, edge_weights)

        g_embed = torch.mean(h, dim=0)
        g_embed = self.readout(g_embed)
        return g_embed


class SceneTrendTransformer(nn.Module):
    def __init__(self, embed_dim, nhead, num_layers):
        super(SceneTrendTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(embed_dim, embed_dim)

    def forward(self, src_seq):
        output = self.transformer_encoder(src_seq)
        prediction = self.decoder(output[-1])
        return prediction


class RiskEvaluator(nn.Module):
    def __init__(self, input_dim):
        super(RiskEvaluator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, g_embed, g_predict):
        combined = torch.cat((g_embed, g_predict), dim=0)
        cts = self.mlp(combined)
        return cts


class STGBuilder:
    def __init__(self, sigma=2.0, interact_thresh=0.4, device='cpu'):
        self.sigma = sigma
        self.interact_thresh = interact_thresh
        self.device = device
        self.sensitive_keywords = ["血汗钱", "还钱", "维权", "集会", "抗议"]

    def build_graph(self, entities, attributes, keypoints, semantic_texts):
        num_nodes = len(entities)
        if num_nodes == 0:
            return None, None, None

        features = []
        for i in range(num_nodes):
            box = entities[i]['bbox']
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            velocity = [0.0, 0.0]

            kpt_vec = np.zeros(34)
            if i < len(keypoints) and keypoints[i] is not None:
                kpts = np.array(keypoints[i]).flatten()
                if len(kpts) == 34:
                    kpt_vec = kpts

            attr_vec = np.zeros(5)

            node_feat = np.concatenate(([center_x, center_y], velocity, kpt_vec, attr_vec))
            features.append(node_feat)

        x = torch.tensor(np.array(features), dtype=torch.float32).to(self.device)

        adj = torch.zeros((num_nodes, num_nodes)).to(self.device)
        edge_weights = torch.zeros((num_nodes, num_nodes)).to(self.device)

        centers = x[:, :2]
        dists = torch.cdist(centers, centers)

        w_space = torch.exp(- (dists ** 2) / (2 * self.sigma ** 2))

        w_interact = torch.zeros_like(dists)

        w_semantic = torch.zeros_like(dists)

        for i, text_info in enumerate(semantic_texts):
            if any(kw in text_info for kw in self.sensitive_keywords):
                for j in range(num_nodes):
                    if i != j:
                        if dists[i, j] < 200:
                            w_semantic[i, j] = 0.9
                            w_semantic[j, i] = 0.9

        edge_weights = w_space + w_interact + w_semantic
        adj = (edge_weights > 0.1).float()

        return x, adj, edge_weights


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
            'classes': None,
            'agnostic_nms': False,
            'augment': False
        }

        self.opt = SimpleNamespace(**default_values)
        self.device = torch_utils.select_device(self.opt.device)
        self.half = self.device.type != 'cpu'
        self.model = attempt_load(self.opt.weights, map_location=self.device)
        self.imgsz = check_img_size(self.opt.img_size, s=self.model.stride.max())
        if self.half:
            self.model.half()

    def detect_test(self, test_list):
        for i, img in enumerate(test_list):
            im0 = img
            img = letterbox(img, new_shape=self.opt.img_size)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            if i == 0:
                batch_img = img
            else:
                batch_img = torch.cat([batch_img, img], axis=0)

        pred = self.model(batch_img, augment=self.opt.augment)[0]
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                   agnostic=self.opt.agnostic_nms)

        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        batch_results = []
        for i, det in enumerate(pred):
            results = []
            if det is not None and len(det):
                det[:, :4] = scale_coords(batch_img.shape[2:], det[:, :4], im0.shape).round()
                det = det.data.cpu().numpy()
                for *xyxy, conf, cls in det:
                    result = {'bbox': xyxy, 'label': names[int(cls)], 'conf': conf}
                    results.append(result)
            batch_results.append(results)
        return batch_results


def getinfo(caid):
    csv_file_path = 'C:\\Users\\19025\\Desktop\\crowd_vis-mainhbver\\crowd_vis-main\\pp-human\\pipeline\\govinf.csv'
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['唯一标识码'] == caid:
                qwe = row['\ufeff通道名称']
                asd = row['经度']
                zxc = row['纬度']
                rty = row['类别']
                return qwe, asd, zxc, rty


def extract_clothing_histogram(image, color_space='hsv'):
    if color_space == 'hsv':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def compare_histograms(hist1, hist2, method=cv2.HISTCMP_CORREL):
    return cv2.compareHist(hist1, hist2, method)


class ImageWindow(QWidget):
    def __init__(self, frame_rgb, humannum, humannum2):
        super().__init__()
        self.setWindowTitle('PyQt5 Image Display')
        self.setGeometry(100, 100, 800, 800)
        humannum += " 预警人数："
        humannum += humannum2
        imagepathhh = "E:\\apache-tomcat-9.0.52\\webapps\\ROOT\\tupian\\"
        imagepath = imagepathhh + str(time.localtime()) + ".jpg"
        frame_rgb.save(imagepath)
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(800, 500)
        self.image_label.move(0, 0)
        self.label = QLabel(self)
        self.label.setText(humannum)
        font = QFont("黑体", 18)
        self.label.setFont(font)
        self.label.setFixedSize(600, 100)
        self.label.move(50, 600)
        frame_rgb = QImage(frame_rgb)
        pixmap = QPixmap.fromImage(frame_rgb)
        self.image_label.setPixmap(pixmap)
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)


class Pipeline(object):
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
        if self.multi_camera:
            self.predictor = []
            for name in self.input:
                predictor_item = PipePredictor(
                    args, cfg, is_video=True, multi_camera=True)
                predictor_item.set_file_name(name)
                self.predictor.append(predictor_item)
        else:
            self.predictor = PipePredictor(args, cfg, self.is_video)
            if self.is_video:
                self.predictor.set_file_name(self.input)

    def _parse_input(self, image_file, image_dir, video_file, video_dir,
                     camera_id, rtsp):
        if image_file is not None or image_dir is not None:
            input = get_test_images(image_dir, image_file)
            self.is_video = False
            self.multi_camera = False
        elif video_file is not None:
            assert os.path.exists(
                video_file
            ) or 'rtsp' in video_file, "video_file not exists and not an rtsp site."
            self.multi_camera = False
            input = video_file
            self.is_video = True
        elif video_dir is not None:
            videof = [os.path.join(video_dir, x) for x in os.listdir(video_dir)]
            if len(videof) > 1:
                self.multi_camera = True
                videof.sort()
                input = videof
            else:
                input = videof[0]
            self.is_video = True
        elif rtsp is not None:
            if len(rtsp) > 1:
                rtsp = [rtsp_item for rtsp_item in rtsp if 'rtsp' in rtsp_item]
                self.multi_camera = True
                input = rtsp
            else:
                self.multi_camera = False
                input = rtsp[0]
            self.is_video = True
        elif camera_id != -1:
            self.multi_camera = False
            input = camera_id
            self.is_video = True
        else:
            raise ValueError(
                "Illegal Input, please set one of ['video_file', 'camera_id', 'image_file', 'image_dir']"
            )
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
            elif key == "VEHICLE_PLATE":
                det_model_dir = cfg[key]["det_model_dir"]
                downloaded_det_model_dir = auto_download_model(det_model_dir)
                if downloaded_det_model_dir:
                    det_model_dir = downloaded_det_model_dir
                    cfg[key]["det_model_dir"] = det_model_dir
                rec_model_dir = cfg[key]["rec_model_dir"]
                downloaded_rec_model_dir = auto_download_model(rec_model_dir)
                if downloaded_rec_model_dir:
                    rec_model_dir = downloaded_rec_model_dir
                    cfg[key]["rec_model_dir"] = rec_model_dir
        elif key == "MOT":
            model_dir = cfg[key]["model_dir"]
            downloaded_model_dir = auto_download_model(model_dir)
            if downloaded_model_dir:
                model_dir = downloaded_model_dir
                cfg[key]["model_dir"] = model_dir


def getid():
    aid = []
    csv_file_path = 'C:\\Users\\19025\\Desktop\\crowd_vis-mainhbver\\crowd_vis-main\\pp-human\\pipeline\\govinf.csv'
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        for row in reader:
            aid.append(row[fieldnames[1]])
        return aid


camera_ids = getid()
rtsplinks = [f"rtsp://13.1.4.60:8319/dss/monitor/param?cameraid={cameraid}&substream=1" for cameraid in camera_ids]


class PipePredictor(object):
    def __init__(self, args, cfg, is_video=True, multi_camera=False):
        if args.run_mode == "openvino":
            self.core = ov.Core()
        else:
            self.core = None

        self.with_mot = cfg.get('MOT', False)['enable'] if cfg.get('MOT', False) else False
        self.with_human_attr = cfg.get('ATTR', False)['enable'] if cfg.get('ATTR', False) else False
        self.with_skeleton_action = cfg.get('SKELETON_ACTION', False)['enable'] if cfg.get('SKELETON_ACTION',
                                                                                           False) else False
        self.with_video_action = cfg.get('VIDEO_ACTION', False)['enable'] if cfg.get('VIDEO_ACTION', False) else False
        self.with_idbased_detaction = cfg.get('ID_BASED_DETACTION', False)['enable'] if cfg.get('ID_BASED_DETACTION',
                                                                                                False) else False
        self.with_idbased_clsaction = cfg.get('ID_BASED_CLSACTION', False)['enable'] if cfg.get('ID_BASED_CLSACTION',
                                                                                                False) else False
        self.with_mtmct = cfg.get('REID', False)['enable'] if cfg.get('REID', False) else False
        self.with_vehicleplate = cfg.get('VEHICLE_PLATE', False)['enable'] if cfg.get('VEHICLE_PLATE', False) else False
        self.with_vehicle_attr = cfg.get('VEHICLE_ATTR', False)['enable'] if cfg.get('VEHICLE_ATTR', False) else False

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

        get_model_dir(self.cfg)

        if self.heatmap:
            self.heatmap_detector = CSRNet_predictor(
                os.path.join(os.path.dirname(__file__), "model", "CSRNet", "model.onnx"), self.core)

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
            self.vehicle_attr_predictor = VehicleAttr.init_with_cfg(args, vehicleattr_cfg)

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
                self.det_action_predictor = DetActionRecognizer.init_with_cfg(args, idbased_detaction_cfg,
                                                                              core=self.core)
                self.det_action_visual_helper = ActionVisualHelper(1)

            if self.with_idbased_clsaction:
                idbased_clsaction_cfg = self.cfg['ID_BASED_CLSACTION']
                basemode = self.basemode['ID_BASED_CLSACTION']
                self.modebase[basemode] = True
                self.cls_action_predictor = ClsActionRecognizer.init_with_cfg(args, idbased_clsaction_cfg,
                                                                              core=self.core)
                self.cls_action_visual_helper = ActionVisualHelper(1)

            if self.with_skeleton_action:
                skeleton_action_cfg = self.cfg['SKELETON_ACTION']
                display_frames = skeleton_action_cfg['display_frames']
                self.coord_size = skeleton_action_cfg['coord_size']
                basemode = self.basemode['SKELETON_ACTION']
                self.modebase[basemode] = True
                skeleton_action_frames = skeleton_action_cfg['max_frames']
                self.skeleton_action_predictor = SkeletonActionRecognizer.init_with_cfg(args, skeleton_action_cfg,
                                                                                        core=self.core)
                self.skeleton_action_visual_helper = ActionVisualHelper(display_frames)
                kpt_cfg = self.cfg['KPT']
                kpt_model_dir = kpt_cfg['model_dir']
                kpt_batch_size = kpt_cfg['batch_size']
                self.kpt_predictor = KeyPointDetector(
                    kpt_model_dir, args.device, args.run_mode, kpt_batch_size,
                    args.trt_min_shape, args.trt_max_shape, args.trt_opt_shape,
                    args.trt_calib_mode, args.cpu_threads, args.enable_mkldnn,
                    use_dark=False, core=self.core)
                self.kpt_buff = KeyPointBuff(skeleton_action_frames)

            if self.with_mtmct:
                reid_cfg = self.cfg['REID']
                basemode = self.basemode['REID']
                self.modebase[basemode] = True
                self.reid_predictor = ReID.init_with_cfg(args, reid_cfg)

            if self.with_mot or self.modebase["idbased"] or self.modebase["skeletonbased"]:
                mot_cfg = self.cfg['MOT']
                model_dir = mot_cfg['model_dir']
                tracker_config = mot_cfg['tracker_config']
                batch_size = mot_cfg['batch_size']
                skip_frame_num = mot_cfg.get('skip_frame_num', -1)
                basemode = self.basemode['MOT']
                self.modebase[basemode] = True
                self.mot_predictor = SDE_Detector(
                    model_dir, tracker_config, args.device, args.run_mode, batch_size,
                    args.trt_min_shape, args.trt_max_shape, args.trt_opt_shape,
                    args.trt_calib_mode, args.cpu_threads, args.enable_mkldnn,
                    skip_frame_num=skip_frame_num, draw_center_traj=self.draw_center_traj,
                    secs_interval=self.secs_interval, do_entrance_counting=self.do_entrance_counting,
                    do_break_in_counting=self.do_break_in_counting, region_type=self.region_type,
                    region_polygon=self.region_polygon, core=self.core)

            if self.with_video_action:
                video_action_cfg = self.cfg['VIDEO_ACTION']
                basemode = self.basemode['VIDEO_ACTION']
                self.modebase[basemode] = True
                self.video_action_predictor = VideoActionRecognizer.init_with_cfg(args, video_action_cfg,
                                                                                  core=self.core)

        self.stg_builder = STGBuilder()
        self.vtan = VTAN(input_dim=43, hidden_dim=64, output_dim=128)
        self.scene_predictor = SceneTrendTransformer(embed_dim=128, nhead=4, num_layers=2)
        self.risk_evaluator = RiskEvaluator(input_dim=128)
        self.embedding_buffer = deque(maxlen=20)
        self.cts_history = deque(maxlen=30)
        self.t_value = 0.6
        self.t_trend = 0.2

    def set_file_name(self, path):
        if path is not None:
            self.file_name = os.path.split(path)[-1]
            if "." in self.file_name:
                self.file_name = self.file_name.split(".")[-2]
        else:
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
        batch_loop_cnt = math.ceil(float(len(input)) / self.det_predictor.batch_size)
        for i in range(batch_loop_cnt):
            start_index = i * self.det_predictor.batch_size
            end_index = min((i + 1) * self.det_predictor.batch_size, len(input))
            batch_file = input[start_index:end_index]
            batch_input = [decode_image(f, {})[0] for f in batch_file]

            if i > self.warmup_frame:
                self.pipe_timer.total_time.start()
                self.pipe_timer.module_time['det'].start()
            det_res = self.det_predictor.predict_image(batch_input, visual=False)
            det_res = self.det_predictor.filter_box(det_res, self.cfg['crop_thresh'])
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
                    attr_res = self.attr_predictor.predict_image(crop_input, visual=False)
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
                    attr_res = self.vehicle_attr_predictor.predict_image(crop_input, visual=False)
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
                    platelicense = self.vehicleplate_detector.get_platelicense(crop_input)
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
        cameraindex = 0
        if setlocal == 0:
            capture = cv2.VideoCapture(rtsplinks[cameraindex], cv2.CAP_FFMPEG)
        else:
            capture = cv2.VideoCapture(0)

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if len(self.pushurl) > 0:
            pushstream = PushStream()
            pushstream.initcmd(fps, width, height)
        elif self.cfg['visual']:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            out_path = os.path.join(self.output_dir, time.strftime("%Y%m%d_%H%M%S", time.localtime()) + ".mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, fps - 45, (width, height))
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
                for i in range(0, len(self.region_polygon), 2):
                    entrance.append([self.region_polygon[i], self.region_polygon[i + 1]])
                entrance.append([width, height])
            else:
                raise ValueError("region_type:{} unsupported.".format(self.region_type))

        video_fps = fps
        video_action_imgs = []
        if self.with_video_action:
            short_size = self.cfg["VIDEO_ACTION"]["short_size"]
            scale = ShortSizeScale(short_size)

        object_in_region_info = {}
        illegal_parking_dict = None
        cont = True
        camerastart = time.time()

        while (1):
            if time.time() - camerastart >= 6:
                if cameraindex <= len(rtsplinks) - 2:
                    cameraindex += 1
                else:
                    cameraindex = cameraindex - len(rtsplinks) + 1
                capture.release()
                gc.collect()
                try:
                    if setlocal == 0:
                        capture = cv2.VideoCapture(rtsplinks[cameraindex], cv2.CAP_FFMPEG)
                        capture.set(cv2.CAP_PROP_BUFFERSIZE, 5)
                    else:
                        capture = cv2.VideoCapture(0)
                except:
                    pass
                camerastart = time.time()

            ret, frame = capture.read()
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

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgbbb = frame_rgb

            if frame_id > self.warmup_frame:
                self.pipe_timer.total_time.start()

            if self.heatmap and frame_id % video_fps == 0:
                heatmap_count = self.heatmap_detector.run({"image": copy.deepcopy(frame_rgb)})
                heatmap_count[1] = heatmap_count[1].transpose()[::-1]
                plt.imshow(heatmap_count[1])
                plt.savefig("temp.jpg")

            mot_res = None
            if self.modebase["idbased"] or self.modebase["skeletonbased"]:
                if frame_id > self.warmup_frame:
                    self.pipe_timer.module_time['mot'].start()
                mot_skip_frame_num = self.mot_predictor.skip_frame_num
                reuse_det_result = False
                if mot_skip_frame_num > 1 and frame_id > 0 and frame_id % mot_skip_frame_num > 0:
                    reuse_det_result = False

                if frame_id % 10 == 0:
                    res = self.mot_predictor.predict_image([deepcopy(frame)], visual=False,
                                                           reuse_det_result=reuse_det_result)
                    res1 = frame_rgbbb

                mot_res = parse_mot_res(res)
                if frame_id > self.warmup_frame:
                    self.pipe_timer.module_time['mot'].end()
                    self.pipe_timer.track_num += len(mot_res['boxes'])

                if frame_id % 10 == 0 and len(mot_res['boxes']) > 0:
                    entities_for_stg = []
                    for i in range(len(mot_res['boxes'])):
                        box_info = {
                            'id': mot_res['boxes'][i][0],
                            'bbox': mot_res['boxes'][i][3:7]
                        }
                        entities_for_stg.append(box_info)

                    keypoints_for_stg = []
                    if self.with_skeleton_action:
                        crop_input, new_bboxes, ori_bboxes = crop_image_with_mot(frame_rgb, mot_res)
                        kpt_pred = self.kpt_predictor.predict_image(crop_input, visual=False)
                        keypoint_vector, _ = translate_to_ori_images(kpt_pred, np.array(new_bboxes))
                        keypoints_for_stg = keypoint_vector.tolist()

                    attr_for_stg = []

                    ocr = PaddleOCR(use_angle_cls=True, lang="ch")
                    ocr_result = ocr.ocr(frame, cls=True)
                    text_for_stg = []
                    if ocr_result != [None]:
                        for line in ocr_result:
                            for iii in line:
                                text_for_stg.append(str(iii[1][0]))

                    x, adj, edge_weights = self.stg_builder.build_graph(entities_for_stg, attr_for_stg,
                                                                        keypoints_for_stg, text_for_stg)

                    if x is not None:
                        g_embed = self.vtan(x, adj, edge_weights)
                        self.embedding_buffer.append(g_embed)

                        if len(self.embedding_buffer) >= 5:
                            input_seq = torch.stack(list(self.embedding_buffer))
                            g_predict = self.scene_predictor(input_seq)
                            cts_score = self.risk_evaluator(g_embed, g_predict).item()

                            self.cts_history.append(cts_score)

                            trend = 0
                            if len(self.cts_history) > 5:
                                recent = list(self.cts_history)[-5:]
                                trend = (recent[-1] - recent[0]) / 5.0

                            if cts_score > self.t_value and trend > self.t_trend:
                                print(f"ALERT: High Risk Detected! CTS: {cts_score:.2f}, Trend: {trend:.2f}")
                                iii = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                                location, jingdu, weidu, classl = getinfo(camera_ids[cameraindex])
                                iiii = iii + location
                                q_image = QImage(frame_rgb.data.tobytes(), width, height, width * 3,
                                                 QImage.Format_RGB888)
                                imagepathhh = "C:\\Users\\19025\Desktop\\apache-tomcat-9.0.52\\webapps\\ROOT\\tupian\\"
                                imagepath = imagepathhh + time.strftime("%Y%m%d_%H%M%S", time.localtime()) + ".jpg"
                                q_image.save(imagepath)

                                db_obj1 = address(address=location, shoottime=iiii,
                                                  id=random.randint(1, 100000), imagepicpath=imagepath,
                                                  videooopath=imagepath,
                                                  state="0", num=len(mot_res['boxes']),
                                                  recording={"1": len(mot_res['boxes'])},
                                                  banner="High Risk Scene",
                                                  mourn=1, jd=jingdu, wd=weidu,
                                                  cameraindex=camera_ids[cameraindex])
                                db_obj1.save()

            if mot_res is not None and len(mot_res['boxes']) > 0:
                self.pipeline_res.update(mot_res, 'mot')
                crop_input, new_bboxes, ori_bboxes = crop_image_with_mot(frame_rgb, mot_res)

                if self.with_vehicleplate and frame_id % 10 == 0:
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['vehicleplate'].start()
                    plate_input, _, _ = crop_image_with_mot(frame_rgb, mot_res, expand=False)
                    platelicense = self.vehicleplate_detector.get_platelicense(plate_input)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['vehicleplate'].end()
                    self.pipeline_res.update(platelicense, 'vehicleplate')
                else:
                    self.pipeline_res.clear('vehicleplate')

                if self.with_human_attr:
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['attr'].start()
                    attr_res = self.attr_predictor.predict_image(crop_input, visual=False)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['attr'].end()
                    self.pipeline_res.update(attr_res, 'attr')

                if self.with_vehicle_attr:
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['vehicle_attr'].start()
                    attr_res = self.vehicle_attr_predictor.predict_image(crop_input, visual=False)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['vehicle_attr'].end()
                    self.pipeline_res.update(attr_res, 'vehicle_attr')

                if self.with_skeleton_action:
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['kpt'].start()
                    kpt_pred = self.kpt_predictor.predict_image(crop_input, visual=False)
                    keypoint_vector, score_vector = translate_to_ori_images(kpt_pred, np.array(new_bboxes))
                    kpt_res = {}
                    kpt_res['keypoint'] = [keypoint_vector.tolist(), score_vector.tolist()] if len(
                        keypoint_vector) > 0 else [[], []]
                    kpt_res['bbox'] = ori_bboxes
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['kpt'].end()

                    self.pipeline_res.update(kpt_res, 'kpt')
                    self.kpt_buff.update(kpt_res, mot_res)
                    state = self.kpt_buff.get_state()

                    skeleton_action_res = {}
                    if state:
                        if frame_id > self.warmup_frame:
                            self.pipe_timer.module_time['skeleton_action'].start()
                        collected_keypoint = self.kpt_buff.get_collected_keypoint()
                        skeleton_action_input = parse_mot_keypoint(collected_keypoint, self.coord_size)
                        skeleton_action_res = self.skeleton_action_predictor.predict_skeleton_with_mot(
                            skeleton_action_input)
                        if frame_id > self.warmup_frame:
                            self.pipe_timer.module_time['skeleton_action'].end()
                        self.pipeline_res.update(skeleton_action_res, 'skeleton_action')

            if self.with_video_action:
                frame_len = self.cfg["VIDEO_ACTION"]["frame_len"]
                sample_freq = self.cfg["VIDEO_ACTION"]["sample_freq"]
                if sample_freq * frame_len > frame_count:
                    sample_freq = int(frame_count / frame_len)
                if frame_id > self.warmup_frame:
                    self.pipe_timer.module_time['video_action'].start()
                if frame_id % 20 == 0:
                    scaled_img = scale(frame_rgb)
                    video_action_imgs.append(scaled_img)
                if len(video_action_imgs) == frame_len:
                    classes, scores = self.video_action_predictor.predict(video_action_imgs)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['video_action'].end()
                    video_action_res = {"class": classes[0], "score": scores[0]}
                    self.pipeline_res.update(video_action_res, 'video_action')
                    video_action_imgs.clear()

            self.collector.append(frame_id, self.pipeline_res)
            if frame_id > self.warmup_frame:
                self.pipe_timer.img_num += 1
                self.pipe_timer.total_time.end()
            frame_id += 1

            if self.cfg['visual']:
                _, _, fps = self.pipe_timer.get_total_time()
                im = self.visualize_video(frame, self.pipeline_res, self.collector, frame_id, fps,
                                          entrance, records, center_traj,
                                          self.illegal_parking_time != -1,
                                          illegal_parking_dict)
                if self.heatmap:
                    heat_img = cv2.imread("temp.jpg")
                    img2 = cv2.resize(heat_img, (int(heat_img.shape[0] / 1.5), int(heat_img.shape[1] / 1.5)))
                    rows, cols, channels = img2.shape
                    roi = im[0:rows, 0:cols]
                    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                    ret, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY)
                    mask_inv = cv2.bitwise_not(mask)
                    img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
                    img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)
                    dst = cv2.add(img1_bg, img2_fg)
                    im[0:rows, 0:cols] = dst

                show_im = cv2.resize(im, (0, 0), fx=0.5, fy=0.5)
                if return_im:
                    yield {"im": show_im}

                if len(self.pushurl) > 0:
                    pushstream.pipe.stdin.write(im.tobytes())
                else:
                    writer.write(im)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        if self.cfg['visual'] and len(self.pushurl) == 0:
            writer.release()

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
                det_res_i['boxes'] = det_res['boxes'][start_idx:start_idx + boxes_num_i, :]
                im = visualize_box_mask(
                    im,
                    det_res_i,
                    labels=['target'],
                    threshold=self.cfg['crop_thresh'])
                im = np.ascontiguousarray(np.copy(im))
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            if human_attr_res is not None:
                human_attr_res_i = human_attr_res['output'][start_idx:start_idx + boxes_num_i]
                im = visualize_attr(im, human_attr_res_i, det_res_i['boxes'])
            if vehicle_attr_res is not None:
                vehicle_attr_res_i = vehicle_attr_res['output'][start_idx:start_idx + boxes_num_i]
                im = visualize_attr(im, vehicle_attr_res_i, det_res_i['boxes'])
            if vehicleplate_res is not None:
                plates = vehicleplate_res['vehicleplate']
                det_res_i['boxes'][:, 4:6] = det_res_i['boxes'][:, 4:6] - det_res_i['boxes'][:, 2:4]
                im = visualize_vehicleplate(im, plates, det_res_i['boxes'])

            img_name = os.path.split(im_file)[-1]
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            out_path = os.path.join(self.output_dir, img_name)
            cv2.imwrite(out_path, im)
            start_idx += boxes_num_i


def main():
    cfg = merge_cfg(FLAGS)
    print_arguments(cfg)
    pipeline = Pipeline(FLAGS, cfg)
    pipeline.run_multithreads()


def main_new(FLAGS, video_file):
    cfg = merge_cfg(FLAGS)
    print_arguments(cfg)
    pipeline = PipePredictor(FLAGS, cfg)
    pipeline.set_file_name(video_file)
    res = pipeline.predict_video(video_file, return_im=True)
    for i in res:
        yield i


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU'], "device should be CPU, GPU or XPU"
    main()
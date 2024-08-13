import numpy as np
import colorsys
import torch
import sort
import cv2
import time

from PIL import Image
from PIL import Image, ImageDraw, ImageFont
from utils.anchors import get_anchors
from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)
from utils.utils_bbox import BBoxUtility
from openvino.runtime import Model, Core

class OV_Infer():
    def __init__(self, **kwargs):
        self._defaults = {
            #-------------------------------
            #   模型路径
            #-------------------------------
            "model_path" : "ssd.onnx",

            #-------------------------------
            #   模型类别
            #-------------------------------
            "classes_path" : "model_data/voc_classes.txt",

            #-------------------------------
            #   是否开启sort目标跟踪
            #-------------------------------
            "track" : True,

            # -------------------------------
            #   backbone的输入尺寸
            # -------------------------------
            "input_shape" : [300, 300],

            #-------------------------------
            #   先验框的尺寸
            #-------------------------------
            'anchors_size' : [30, 60, 111, 162, 213, 264, 315],
            # -------------------------------
            #   是否对输入图像进行分辨率的保持（不失真）
            # -------------------------------
            'letterbox' : False,

        }
        # -------------------------------
        #   遍历设置好的字典的属性给到类属性  self._defaults--->self
        # -------------------------------
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        # -------------------------------
        #   设置类别
        # -------------------------------
        self.class_names, self.class_nums = get_classes(self.classes_path)

        # -------------------------------
        #   设置anchors,backbone取决于训练时的选择，选择了vgg那就“vgg”，选择了resnet18那就”resnet18“
        #   解析推理的结果时要用，因为ssd的output是根据先验框的相对位移来解码的
        # -------------------------------
        self.anchors = torch.from_numpy(get_anchors(input_shape=self.input_shape, anchors_size=self.anchors_size, backbone="vgg")).type(torch.FloatTensor)

        # if self.cuda:
        #     self.anchors = self.anchors.cuda()
        self.class_nums = self.class_nums + 1

        # -------------------------------
        #   设置不同框的颜色
        # -------------------------------
        hsv_tuples  = [(x / self.class_nums, 1., 1.) for x in range(self.class_nums)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.box_util      = BBoxUtility(self.class_nums)
        self.Initial_model()

    def Initial_model(self):
        # ------------------------------
        #   实例化一个core，用于openvino推理
        # ------------------------------
        self.core = Core()

        # ------------------------------
        #   实例化一个ov的model对象
        # ------------------------------
        self.model = self.core.read_model(self.model_path)

        # ------------------------------
        #   实例化一个ov的编译模型
        # ------------------------------
        self.compile_model = self.core.compile_model(self.model, 'CPU')

        # ------------------------------
        #   如果开启了sort跟踪算法，实例化一个track对象
        # ------------------------------
        if self.track:
            self.mot_tracker = sort.Sort(max_age=1, min_hits=3, iou_threshold=0.3)
        else:
            self.mot_tracker = None

    def preprocess(self, image):
        # ------------------------------
        #   获取图像的shape
        # ------------------------------
        self.image_shape = np.array(np.shape(image)[0:2])

        # ------------------------------
        #   将输入图像转成RGB格式
        # ------------------------------
        image_data = cvtColor(image)

        # ------------------------------
        #   将不同尺寸的输入图像resize到网络的输入尺寸，即300 * 300
        # ------------------------------
        image_data = resize_image(image_data, (self.input_shape[1], self.input_shape[0]), self.letterbox)

        # ------------------------------
        #   图像像素减去均值
        # ------------------------------
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2,0,1)), 0)

        return image_data

    def detect(self, image, crop = False, count = False):
        # ------------------------------
        #   前处理
        # ------------------------------
        preprocessed_input = self.preprocess(image)

        # ------------------------------
        #   转化成torch的形式
        # ------------------------------
        preprocessed_input = torch.from_numpy(preprocessed_input).type(torch.FloatTensor)
        # if self.cuda:
        #     preprocessed_input = preprocessed_input.cuda()

        # ------------------------------
        #   前向传播
        # ------------------------------
        ov_outputs = self.compile_model(preprocessed_input)
        keys = ov_outputs.keys()
        ov_det, ov_cls = 0, 0
        for i, key in enumerate(keys):
            if i == 0:
                ov_det     = torch.from_numpy(ov_outputs[key])
            else:
                ov_cls     = torch.from_numpy(ov_outputs[key])
        outputs = (ov_det, ov_cls)
        results = self.box_util.decode_box(outputs, self.anchors, self.image_shape, self.input_shape,
                                           self.letterbox, nms_iou=0.45, confidence=0.5)

        # ------------------------------
        #   如果没有检测到物体 返回原图
        # ------------------------------
        if len(results[0]) <= 0:
            return image

        top_label = np.array(results[0][:,4], dtype='int32')
        top_conf  = results[0][:, 5]
        top_boxes = results[0][:, :4]

        # ---------------------------------------------------------#
        #   sort需要：torch.tensor的切片操作：跳过中间的元素，即不要类别索引
        # ---------------------------------------------------------#
        indices = torch.tensor([0, 1, 2, 3, 5])
        top_sort = results[0][:, indices]

        # ---------------------------------------------------------#
        #   如果开启了sort跟踪
        # ---------------------------------------------------------#
        if self.mot_tracker != None and self.track == True:
            trackers         = self.mot_tracker.update(top_sort)

        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)

        # ---------------------------------------------------------#
        #   图像绘制
        # ---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[top_label[len(top_label) - i - 1]]
            if self.mot_tracker == None:
                box = top_boxes[i]
                score = top_conf[i]

            '''
            使用跟踪结果
            '''
            if self.mot_tracker != None:
                if i >= len(trackers):
                    continue
                box             = trackers[i][:4]
                score           = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            if self.mot_tracker != None:
                label = '{} {:.2f} {}'.format(predicted_class, score, trackers[i][4])
            else:
                label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textbbox((0, 0), label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

if __name__ == "__main__":
    video_path = 0
    capture = cv2.VideoCapture(video_path)
    ref, _ = capture.read()
    if not ref:
        raise ValueError("video path is error!")

    fps = 0.0
    ov_infer = OV_Infer()
    while True:
        t1 = time.time()

        ref, frame = capture.read()
        if not ref:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = Image.fromarray(np.uint8(frame))

        # ------------------------------
        #   开始检测，将PIL的格式转成np格式给cv展示
        # ------------------------------
        frame = np.array(ov_infer.detect(frame))
        # ------------------------------
        #   调整一下通道到bgr给cv
        # ------------------------------
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # ------------------------------
        #   计算fps并且显示在窗口上
        # ------------------------------
        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % (fps))
        frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("video", frame)
        '''
        & 0xff是做了一个二进制的与操作，0xff是一个十六进制的数，对应的二进制是11111111。
        这个操作的目的是只保留返回值的最后八位，因为在某些系统中，cv2.waitKey的返回值不止8位。
        '''
        c = cv2.waitKey(1) & 0xff

        '''
        ASCII码27对应的是ESC按键
        '''
        if c == 27:
            capture.release()
            break



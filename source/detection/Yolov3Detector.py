import torch
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from .Detector import Detector
from data.transformation import Yolov3Transform
from dataclasses import dataclass
from torchvision.transforms import Compose, Lambda, ToTensor
from yolov3.models import Darknet, load_darknet_weights, attempt_download, ONNX_EXPORT
from yolov3.utils.utils import non_max_suppression, scale_coords, plot_one_box

@dataclass
class Yolov3Detector:
    cfg: str = 'cfg/yolov3.cfg'
    weights: str = 'weights/best.pt'
    img_size: tuple = (416, 416)
    transform: callable = None
    classes: dict = None
    device: torch.device = torch.device('cpu')
    output: str = None
    half: bool = False
    view_img: bool = False

    def __post_init__(self):
        self.half = False
        self.transform = self.transform or Compose([
            Yolov3Transform(self.half, self.img_size),
            ToTensor()])

        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]

        self.model = self.get_model()

    def get_model(self):
        """
        Return a Darknet model corretly initialized.
        :return: Darknet
        """
        model = Darknet(self.cfg, self.img_size)
        attempt_download(self.weights)
        if self.weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(self.weights, map_location=self.device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, self.weights)

        model.to(self.device).eval()

        # Half precision
        self.half = self.half and self.device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            model.half()

        return model

    def __call__(self, imgs, conf_thres=0.3, nms_thres=0.5):
        """

        :param imgs: An array of RGB images as numpy arrays.
        :param conf_thres:
        :param nms_thres:
        :return: A list of predictions correctly rescaled.
        """
        preds_imgs = []
        for img in tqdm(imgs):
            x = img.copy()
            if self.transform is not None:
                x = self.transform(x)
            if x.ndimension() == 3:
                x = x.unsqueeze(0)
            with torch.no_grad():
                preds = self.model(x)[0]
                if self.half:
                    preds = preds.float()
                preds = non_max_suppression(preds, conf_thres, nms_thres)
                #  rescale predictions
                preds[:, :4] = scale_coords(x.shape[2:], preds[:, :4], img.shape).round()
                preds_imgs.append(preds)

        return preds_imgs

    def add_bb_on_img(self, img, preds):
        """
        Given and image and predictions obtained from calling the detector, plot the bounding boxes on the img
        :param img:
        :param preds:
        :return:
        """

        for i, pred in enumerate(preds):
            *xyxy, conf, _, cls = pred
            # print(f'class={self.classes[int(cls)]:<10} coords={xyxy}')
            label = '%s %.2f' % (self.classes[int(cls)], conf)
            plot_one_box(xyxy, img, label=label, color=self.colors[int(cls)])

        return img

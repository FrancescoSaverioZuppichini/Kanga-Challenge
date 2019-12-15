import pytesseract
import cv2
from detection import Detector
from PIL import Image
from dataclasses import dataclass
from utils import imshow
import numpy as np
from detection import Detector

@dataclass
class OCRDetector(Detector):
    """
    Easy peasy ocr with tesseract, to improve the prediction you can pass 
    `text_color` to cutoff all values that are not text. 

    TODO: probably there is a cv2 function in which we can define a color and a range.
    """
    text_color: int = 165
    smooth: bool = True # if True apply Gaussian Blur to remove noise
    show_img: bool = False
    transform: callable = None
    config: str = ''
    def detect(self, imgs):
        texts = []
        for img in imgs:
            if self.smooth: 
                img = cv2.GaussianBlur(img, (3,3), 0)
            if self.text_color:
                img = img < self.text_color
            if self.transform is not None: img = self.transform(img)
            img = img.astype('uint8')
            if self.show_img : imshow(img)
            x = Image.fromarray(img * 255)
#             x.show()
            text = pytesseract.image_to_string(x, lang='eng', config=self.config)
            texts.append(text)
            
        return texts
import threading
import matplotlib.pyplot as plt
import re
from multiprocessing import Queue
from dataclasses import dataclass
from detection.Detection import ObjectDetection

def imshow(x):
    fig = plt.figure()
    plt.imshow(x)
    plt.show()
    return fig


@dataclass
class ThreadScheduler:
    """
    Do something non blocking and store the result in a multipressing queue.

    TODO: maybe there is some problem, e.g. in jupyter if the cell where ThreadScheduler is instantiace is run more than once the output is weird.
    """
    target: callable
    queue: Queue = Queue()
    th: threading.Thread = None

    def __call__(self, *args, **kwargs):
        if self.th is None or not self.th.isAlive():
            self.th = threading.Thread(target=self.target_wrapper,
                                       args=args,
                                       kwargs=kwargs)
            self.th.start()

    def target_wrapper(self, *args, **kwargs):
        res = self.target(*args, **kwargs)
        print(res)
        self.queue.put(res)


@dataclass
class DetectionPerClass:
    detectors : dict = None
        
    def __call__(self, detections: [ObjectDetection]):
        return [ {det.cls: self.detectors[det.cls](det.crop)} for det in detections  if det.cls in self.detectors]
 

def extract_only_numbers(text):
    """
    Little function to extract only numbers from a string. Usuful in OCR if we want to detect only digits.
    Also, in converts 'o's and 'O's to 0 since it appears to be hard for tesseract even if I fucking tell it to 
    parse only digits. GG google.
    
    :param text: A string
    :type text: str
    :return: string with only the digits
    :rtype: str
    """
    text = text.replace('O', '0')
    text = text.replace('o', '0')
    text = re.findall(r'\d+', text)
    return ''.join(text)

def crops_from_yolov3_preds(preds, src):
        for det in preds:
            *coord, conf, _, cls = det
            coord = np.array(coord).astype(np.int)
            #             crop out from src image
            x1, y1, x2, y2 = coord
            crop = src[y1:y2, x1:x2]
            yield crop

def crops_from_df_preds(preds, src):
        for i, det in preds.iterrows():
            crop = src[int(det.y):int(det.y2), int(det.x):int(det.x2)]
            yield crop
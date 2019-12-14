from abc import ABC, abstractmethod

class Detector:
    """
    Basic Detector infarface
    """
    def __call__(self, frames: list = [], *args, **kwargs) -> dict:
        res = self.detect(frames, *args, **kwargs)
        return res

    @abstractmethod
    def detect(self, images: list = [], *args, **kwargs) -> dict:
        pass

    @abstractmethod
    def to_JSON() -> dict:
        pass
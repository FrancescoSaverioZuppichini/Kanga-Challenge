from abc import ABC,abstractmethod 

class Detector:
    def __call__(self, frames: list = [], *args, **kwargs) -> dict:
        res = self.detect(frames, *args, **kwargs)
        return res

    @abstractmethod
    def detect(self, frames: list = [], *args, **kwargs) -> dict:
        pass

    @abstractmethod
    def to_JSON() -> dict:
        pass
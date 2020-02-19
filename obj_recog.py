from ialab.input.image import CameraInput
from ialab.pipeline.model import RegionInPaintProcessor, parameter_transformer
from ialab.pipeline.detection import YoloV3, Skipper
from ialab.pipeline.output import ShowImage
from os import path

if __name__ == '__main__':
    base_path = '/home/harold/Documents/models/yolov3'
    start = YoloV3(path.join(base_path, 'yolov3.cfg'), path.join(base_path, 'yolov3.weights'),
                   path.join(base_path, 'coco.names'),
                   outputs=[
        RegionInPaintProcessor(outputs=[
            ShowImage('demo')
        ])
    ], skiper=Skipper(4))

    video = CameraInput((800, 600), [start])
    video.run()
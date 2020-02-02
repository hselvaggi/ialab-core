from ialab.input.image import CameraInput
from ialab.pipeline.model import RegionInPaintProcessor
from ialab.pipeline.detection import YoloV3, Skiper
from ialab.pipeline.output import ShowImage


if __name__ == '__main__':
    start = YoloV3('./models/yolov3/yolov3.cfg', './models/yolov3/yolov3.weights',
                   './models/yolov3tiny/coco.names',
                   outputs=[
        RegionInPaintProcessor(outputs=[
            ShowImage('demo')
        ])
    ], skiper=Skiper(4))

    video = CameraInput((800, 600), [start])
    video.run()
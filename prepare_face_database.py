import cv2 as cv
from pipeline.faces import FaceRecognition, FaceExtractor
from input.image import ImageInput
from pipeline.model import RegionInPaintProcessor, Processor
from pipeline.output import ImageWriter, ShowImage


class Waiter(Processor):
    def __init__(self):
        Processor.__init__(self, [])

    def __call__(self, *args, **kwargs):
        cv.waitKey(0)


if __name__ == '__main__':

    p1 = FaceRecognition('./data/faces', [
        FaceExtractor([
            ImageWriter('./data/sample')]
        ),
        RegionInPaintProcessor([ShowImage('face'), Waiter()])
    ])

    video = ImageInput('./data/old', (320, 240), [p1])
    video.run()

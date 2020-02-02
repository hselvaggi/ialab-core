import cv2 as cv
from ialab.pipeline.detection import FaceRecognition, FaceExtractor
from ialab.input.image import ImageInput
from ialab.pipeline.model import RegionInPaintProcessor, KeyWait
from ialab.pipeline.output import ImageWriter, ShowImage


if __name__ == '__main__':

    p1 = FaceRecognition('./data/faces', [
        FaceExtractor([
            ImageWriter('./data/sample')]
        ),
        RegionInPaintProcessor([ShowImage('face'), KeyWait()])
    ])

    video = ImageInput('./data/old', (320, 240), [p1])
    video.run()

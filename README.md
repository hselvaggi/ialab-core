# Introduction

I find myself writting the same code over again every time I want to run some experiment or try
some new IA stuff most of the time loosing focus on the important thing I'm trying to learn
or understand. In order to avoid this pattern I tried to write a small framework that help me
have those standard task already solved and at the same time helps me structure the code
in a readable and maintainable way.

# Installation

This project has been developed with python 3.6. 

```bash
pip3 install ialab-core
```

# Architectural considerations

I will try to explain they why by means of an example in computer vision. 
Most of the time I try some experiment it happens to be as follows:

* I need to read video either from disk or from my notebook camera.
* The image needs to be scaled for performance reasons
* I need to run some sort of algorithm on the image (this may imply one or a secuence of algorithms)
* I want to display the result or optionally display it and send the result to an image or video file

Let's take those steps into a more abstract representation


Read Image -> Scale Image -> Face Detection (to say something specific) -> Display Result
-> Send to disk

This looks very similar to a pipes and filters architecture where I can send the output 
for some step to one or more processors. Nothing new here this is just a graph structure
as it is happening on most new technologies stacks that try to help us building
better and more maintainable software.

Each processing node in this architecture will be called a Processor. 
There are special types of Processors which are the input and output ones. The main
difference between them is that input processors can run and control the execution (this will
change in future versions where a separate executor will appear) and output processor don't try
to offer their output for further processing. On the other hand a regular processor will do
something with the input data and will offer this data for any other processor that
is interested in doing so.

## Example

As an example let's thing you have a folder with images where each image contain somewhere
in it the face of a person and the file is named as the person. You want to do face detection
using a siamesse network but first you want to extract the faces from the images in order to
reduce the memory footprint, loading time and the size of the application you deliver
to your customer.

To do this you can write down the followin short application using our library.

```python
graph = FaceRecognition(original_pictures_path, [
        FaceExtractor([
            ImageWriter(face_picture_path)]
        )
    ])

video = ImageInput(original_pictures_path, (320, 240), [graph])
video.run()
```

## Running the examples

### YoloV3

To run YoloV3 you need to download the configuration files and place them into 
./models/yolov3. There are 3 files that need to be downloaded yolov3.cfg, yolov3.weights
and coco.names. These files can be downloaded from https://pjreddie.com/darknet/yolo/.

To avoid manual download of the model files execute download_models.sh and it will prepare
the models/yolov3 folder for you.

After downloading just execute python3 obj_recog.py
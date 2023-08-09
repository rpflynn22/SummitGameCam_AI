# Summit Lake Game Camera AI Program
**Author**: Keane Flynn\
**Organization**: Summit Lake Paiute Tribe\
**Date**: 08/09/2023\
**Contact**: keaneflynn@nevada.unr.edu

![Mountain Lions](https://github.com/SummitLakeNRD/SummitGameCam_AI/blob/main/test_images/mountainLion.jpg)
*Mama and two cubs seen at one of our trail cameras*

## Description
This program was designed to leverage Ultralytics's YOLOv8 image classification to expedite SLPT's game camera image processing. It is acknowledged that it
will not be perfect so some human supervision is necessary to classify images that are deemed low-confidence inferences. The neural network file is located
within the yolov8 directory and has been trained using images sourced from iNaturalists public API. Future updates to the model will incorporate SLPT's
image data it has gathered from the game cameras. For any questions about the program, feel free to contact Keane Flynn at the above email address.

## Hardware
This program currently is operating specifically to Browning trail cameras. This is due to the exif data indexing for date/time, can work for any images if 
this data is not important to you or you adjust it for your specific use case (see [lines of code](https://github.com/SummitLakeNRD/SummitGameCam_AI/blob/9b79a89aae1f57f8c91cb52f7e959ddc2156e757/main.py#L57-L60)). 
To expedite processing images, it is best to run this program on a computer with an Nvidia chipset with CUDA & CUDNN. Then install PyTorch and Ultralytics
will allow you to reduce processing latency. This is not necessary, however, as it is built in with an option to run on your CPU if no CUDA enabled GPU is
detected upon program launch.

## Program Prerequisites
I am going to explain this in terms a biologist can understand, so my apologies for this explanation being the equivalent of a Fisher Price "My First Hello 
World" Program. You will need to [install python](https://www.python.org/downloads/) and pip which should come with the download linked here and git which
can be [downloaded here for windows](https://git-scm.com/download/win). Open a terminal (terminal, powershell, cmd, etc) and [migrate](https://adamtheautomator.com/powershell-change-directory/) to your desktop directory. 
From here, click the green 'code' button at the top of this page and copy the HTTPS code link provided. Then issue the following command in your terminal:
`git clone https://github.com/SummitLakeNRD/SummitGameCam_AI.git`
Once completed, you will change directories into the SummitGameCam_AI directory you just created and issue the following command:
`pip install -r requirements`
This will install all of the necessary prerequisites for you and you will be ready to process images.

# Running the Program
To obtain the positional and conditional arguments for the python program, issue the following command:
`python main.py -h`
To run a sample code from this repository issue the following command:
`python main.py test_images/ yolov8/gameCamModel-yolov8l-cls.pt outputFiles SampleSite --conf_threshold 0.65`

# Input and Output
### Inputs
- input_directory: simply a directory containing all of your images to be processed
- model_file: path to a YOLOv8 model file that has a .pt file extension
- output_directory: the name of your desired output directory (ex. gameCam_1)
- site_name: name of site to be placed into output text file (ex. upperWatershedSite_1)

### Output
Upon completion, the program will spit out a directory with a subdirectory for each class present in your YOLOv8 model as well as one subdirectory labeled 
'forReview' for human-supervised classifying. Each subdirectory will contain images that met the confidence threshold for classification as well as their
supplementing json file containing the date/time the image was taken, site name, species ID, and confidence value for inference. The 'forReview' folder will
contain all images that did not meet the inference confidence threshold. By default these images do not have json files with them, however you can uncomment
[this code block](https://github.com/SummitLakeNRD/SummitGameCam_AI/blob/9b79a89aae1f57f8c91cb52f7e959ddc2156e757/main.py#L72-L84) to turn that function on.
Cheers.

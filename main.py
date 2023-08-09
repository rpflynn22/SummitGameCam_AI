from argparse import ArgumentParser
from ultralytics import YOLO
import json
import os
import shutil
import PIL.Image
from torch import cuda


def main():
    parser = ArgumentParser(description='Summit Game Cam AI')
    parser.add_argument('input_directory', type=str, help='path/to/game/cam/images/')
    parser.add_argument('model_file', type=str, help='path/to/file/yolo.pt')
    parser.add_arguemnt('output_directory', type=str, help='output directory name')
    parser.add_argument('site_name', type=str, help='site name')
    parser.add_argument('--conf_threshold', type=float, default=0.7, help='positional argument to determine minimum threshold for confirmed inference')
    args = parser.parse_args()

    ###Load neural network file###
    try:
        model = YOLO(args.model_file)
    except:
        print('Error loading neural network, verify it is a YOLOv8 file and the file path is correct')
        exit(1)
    classes = model.model.names

    ###Generate input/output directories###
    cwd = os.getcwd()
    inputDir = cwd + '\\' + args.input_directory # CHANGE FOR NOT CURRENT DIRECTORY
    outputDir = cwd + '\\' + args.output_directory # main directory
    os.makedirs(outputDir, exist_ok=True)

    for subject in classes: # subdirectories for each species
        subjectDir = outputDir + '\\' + classes[subject]
        os.makedirs(subjectDir)
    os.makedirs(output + '\\forReview') # subdirectory for non-confident predictions for human review

    ###Check for CUDA capable device###
    if cuda.device_count() is not 0: # verify graphics card to set processor location (gpu vs cpu)
        proc = 0
    else
        proc = cpu

    ###Loop through source directory images and perform inference
    for filename in os.listdir(inputDir):
        if filename.endswith('.JPG'):
            results = model(source=filename, verbose=False, 
                            conf=args.conf_threshold, device=proc)
            probs = results[0].probs
            predConf = probs.top1conf.tolist()
            predClass = probs.top1
            if predConf > args.conf_threshold:
                exif = PIL.Image.open(filename)._getexif()
                date_time = exif[306] # Don't ask me why but our browning game cams index their exif data as #306 being the date/time, idk bro
                json_filename = filename.split('.')[0]
                json_file = {
                    "dateTime": str(date_time),
                    "site": args.site_name,
                    "species": classes[predClass]
                    "confidence": predConf
                }
                with open("{}\\{}\\{}.json".format(cwd, classes[predClass], json_filename), 'w') as f:
                    json.dump(json_file, f)
                shutil.copy(filename, "{}\\{}\\".format(cwd, classes[predClass]))
            else
                pass # PLACE IMAGE INTO 'forReview' FOLDER

        else
            continue

        #Grab highest confidence name
        #Grab confidence value
        #if below args.conf_threshold; then; place into 'for_review' folder; else place into folder given its name
        #create .json file with site_name, date & time of image capture (yank from exif data?), species ID, confidencce value


if __name__=='__main__':
    main()
from argparse import ArgumentParser
from ultralytics import YOLO
from torch import cuda
import json
import os
import shutil
import PIL.Image


def main():
    parser = ArgumentParser(description='Summit Game Cam AI')
    parser.add_argument('input_directory', type=str, help='path/to/game/cam/images')
    parser.add_argument('model_file', type=str, help='path/to/file/yolo.pt')
    parser.add_argument('output_directory', type=str, help='output directory name')
    parser.add_argument('site_name', type=str, help='site name')
    parser.add_argument('--conf_threshold', type=float, default=0.8, help='positional argument to determine minimum threshold for confirmed inference')
    args = parser.parse_args()

    ###Load neural network file and pull class list###
    try:
        model = YOLO(args.model_file)
    except:
        print('Error loading neural network, verify it is a YOLOv8 file and the file path is correct')
        exit(1)
    classes = model.model.names


    ###Generate output directories###
    cwd = os.getcwd()
    outputDir = os.path.join(cwd, args.output_directory) # main directory
    os.makedirs(outputDir, exist_ok=True)

    for subject in classes: # subdirectories for each species
        subjectDir = os.path.join(outputDir, classes[subject]) 
        os.makedirs(subjectDir, exist_ok=True)
    os.makedirs(os.path.join(outputDir, 'forReview'), exist_ok=True) # subdirectory for non-confident predictions for human review


    ###Check for CUDA capable device###
    if cuda.device_count() is not 0: # verify compatible graphics card to set processor location (gpu vs cpu)
        proc = 0
    else:
        proc = 'cpu'


    ###Loop through source directory images and perform inference
    for file in os.listdir(args.input_directory):
        filename = args.input_directory + file
        if file.lower().endswith('.jpg') or file.lower().endswith('.png') or file.lower().endswith('.jpeg'):
            results = model(source=filename, verbose=False, device=proc) # perform yolov8 classification inference
            probs = results[0].probs
            predConf = probs.top1conf.tolist()
            predClass = probs.top1
            if predConf >= args.conf_threshold:
                exif = PIL.Image.open(filename)._getexif()
                try:
                    dateTime = exif[306] # Don't ask me why but our browning game cams index their exif data as #306 being the date/time, idk bro
                except TypeError:
                    print(str('Image ' + file + ' is not from a browning game camera, please delete this image and restart the program.'))
                    exit(1)
                jsonFilename = file.split('.')[0] + ".json"
                jsonFile = {
                    "dateTime": str(dateTime),
                    "site": args.site_name,
                    "species": classes[predClass],
                    "confidence": predConf
                }
                with open(os.path.join(outputDir, classes[predClass], jsonFilename), 'w') as f:
                    json.dump(jsonFile, f)
                shutil.copy(filename, os.path.join(outputDir, classes[predClass]))
            else:
                ''' # Uncomment to get json data for 'unidentifiable' images
                exif = PIL.Image.open(filename)._getexif()
                dateTime = exif[306] # Don't ask me why but our browning game cams index their exif data as #306 being the date/time, idk bro
                jsonFilename = file.split('.')[0] + ".json"
                jsonFile = {
                    "dateTime": str(dateTime),
                    "site": args.site_name,
                    "species": classes[predClass],
                    "confidence": round(predConf, 4)
                }
                with open(os.path.join(outputDir, 'forReview', jsonFilename), 'w') as f:
                    json.dump(jsonFile, f)
                '''
                shutil.copy(filename, os.path.join(outputDir, 'forReview'))
        else:
            continue
    print('Images processed, see output directory for classified images')


if __name__=='__main__':
    main()
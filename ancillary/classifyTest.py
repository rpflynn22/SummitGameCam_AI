from ultralytics import YOLO
import cv2
from argparse import ArgumentParser
import os


def main():
    parser = ArgumentParser(description='Summit Game Cam AI')
    parser.add_argument('input_directory', type=str, help='path/to/game/cam/images/')
    parser.add_argument('model_file', type=str, help='path/to/file/yolo.pt')
    parser.add_argument('--conf_threshold', type=float, default=0.5, help='positional argument to determine minimum threshold for confirmed inference')
    args = parser.parse_args()

    image = args.input_directory

    model = YOLO(args.model_file)
    classes = model.model.names

    
    results = model(source=image, verbose=False, device=0)

    probs = results[0].probs
    predConf = probs.top1conf.tolist()
    predClass = probs.top1

    #print(predConf)
    print(classes[predClass], ": ", predConf)


    cv2_image = cv2.imread(image)
    cv2_image = cv2.resize(cv2_image, (720,360))
    cv2.imshow('prediction', cv2_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__=='__main__':
    main()
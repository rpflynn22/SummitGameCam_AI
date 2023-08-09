from argparse import ArgumentParser
import os
from PIL import Image

def main():
    parser = ArgumentParser(description='tests images for corrupt ones')
    parser.add_argument('image_directory', type=str, help='path/to/images/')
    args = parser.parse_args()

    for filename in os.listdir(args.image_directory):
        if filename.__contains__('.jpg') or filename.__contains__('.jpeg') or filename.__contains__('.png'):
            try:
                img = Image.open(args.image_directory+filename)  # open the image file
                img.verify()
            except:
                print(filename)

if __name__=='__main__':
    main()
from osgeo import gdal
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

scale = '-scale min_val max_val'
options_list = [
    '-ot Byte',
    '-of JPEG',
    '-b 1',
    '-b 2',
    '-b 3',
    scale
] 
options_string = " ".join(options_list)

gdal.Translate(args["image"][:-3]+'jpg',
               args["image"],
               options=options_string)
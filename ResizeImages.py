from PIL import Image
from glob import glob
import os
def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(imageName)

for i in range(0, 501):
    os.chdir("/Users/aakash/Downloads/Hand-Gesture-Recognition-Using-Background-Elllimination-and-Convolution-Neural-Network-master/Dataset/")
    for file in glob("*/*.png"):
        resizeImage(file)
    


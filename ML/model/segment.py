import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image

def Edge_detection(): # function for outsource usage
    path_data = ""
    img = Image.open(r'C:\Users\Laptop\Desktop\ml-deploy\first_demo\project\ML\data\pvmodule.jpg').convert('L')
    np_array = np.array(img)
    edges = cv.Canny(np_array,50,190)
    pil_image=Image.fromarray(edges)
#pil_image.show()
    pil_image.save(r"C:\Users\Laptop\Desktop\ml-deploy\first_demo\project\ML\output\output.jpg")


# bike
def bike_rental(x, y):
    pass

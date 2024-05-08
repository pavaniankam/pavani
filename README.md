# Draw Bounding boxes and crop the image from the CSV file
# Libraries we have to install for this
import os
import csv
from PIL import Image
# Drawing Boxes on Images
1.It reads the CSV file and opens each image mentioned in the CSV.
2.For each image, it extracts the bounding box coordinates (left, top, right, bottom) and draws rectangles on the image using the ImageDraw module from PIL (Python Imaging Library). The rectangles are drawn around the objects specified by the bounding box coordinates.
```
csv_file = "/home/pavaniankam/Downloads/7622202030987_bounding_box.csv"
image_dir = "/home/pavaniankam/Downloads/7622202030987"
output_dir = "/home/pavaniankam/Downloads/7622202030987_with_boxes"
```
```
os.makedirs(output_dir, exist_ok=True)
def draw_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        left = int(box['left'])
        top = int(box['top'])
        right = int(box['right'])
        bottom = int(box['bottom'])
        draw.rectangle([left, top, right, bottom], outline="red")
    return image
```
# Cropping Images based on Bounding Boxes
1.It also crops the images based on the bounding box coordinates.
2.For each image, it calculates the cropping area defined by the bounding box coordinates and extracts that portion of the image using the crop() method from PIL.
3.Cropped images are saved with a prefix indicating their order in case there are multiple objects in the image.
```
def crop_image(image, boxes):
    cropped_images = []
    for box in boxes:
        left = int(box['left'])
        top = int(box['top'])
        right = int(box['right'])
        bottom = int(box['bottom'])
        cropped_img = image.crop((left, top, right, bottom))
        cropped_images.append(cropped_img)
    return cropped_images
```
# Here's the main parts of the script
1. 'draw_boxes' function: Draws bounding boxes on images using PIL's 'ImageDraw' module.
2. 'crop_image' function: Crops the image based on the bounding box coordinates.

    Main Loop:
1. It iterates through each row in the CSV file.
2. For each row, it opens the corresponding image file.
3. It extracts the bounding box coordinates from the CSV and performs both drawing boxes and cropping images operations.
4. It saves the cropped images with appropriate filenames in the specified output directory.
5. It also saves the original images with drawn bounding boxes.
```
with open(csv_file, 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        image_name = row['filename']
        image_path = os.path.join(image_dir, image_name)
        output_path = os.path.join(output_dir, image_name)
        image = Image.open(image_path)
        boxes = [{'left': row['xmin'], 'top': row['ymin'], 'right': row['xmax'], 'bottom': row['ymax']}]
        cropped_images = crop_image(image, boxes)
        for i, cropped_img in enumerate(cropped_images):
            cropped_img.save(os.path.join(output_dir, f"{i}_{image_name}"))  
        full_image_with_boxes = draw_boxes(image, boxes)
        full_image_with_boxes.save(os.path.join(output_dir, f"full_{image_name}"))
```
Here is the Output of this
/home/pavaniankam/Downloads/7622202030987_with_boxes


## Imagehistogram
````
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
```
These lines import necessary libraries:
1.numpy for numerical operations.
2.cv2 as cv for OpenCV, a library mainly aimed at real-time computer vision.
3.pyplot from matplotlib for plotting graphs.
 ```
img = v.imread('/home/pavaniankam/Desktop/Pavani/blue-cars.jpeg')
```
1.Reads an image file named "blue-cars.jpeg" located at the given file path.
2.cv.imread() loads an image from the specified file path.
```
cv.imwrite("/home/pavaniankam/Desktop/Pavani/histogram.png",img)
```
1.Writes the loaded image to a file named "histogram.png" at the given file path.
2.This line seems to be redundant and unnecessary for histogram calculation or plotting.
```
assert img is not None, "file could not be read, check with os.path.exists()"
```
1.Checks if the image was successfully loaded. If the image is None, it raises an assertion error with the provided message.
```
color = ('b','g','r')
```
1.Defines a tuple containing color channels: blue, green, and red.
```
for i,col in enumerate(color):
```
1.Iterates over the colors in the color tuple along with their corresponding indices.
```
 histr = cv.calcHist([img],[i],None,[256],[0,256])
 plt.plot(histr,color = col)
 plt.xlim([0,256])
plt.show()
```
1.Calculates the histogram for the i-th color channel (blue, green, or red) of the image.
2.cv.calcHist() calculates the histogram of a set of arrays (images) and returns it as a numpy array.
3.Plots the histogram.
4.Sets the x-axis limits of the plot to [0, 256] which corresponds to the range of pixel values.
5.Displays the plot.


```
## Range
```
num = list(range(10))
previousNum = 0
for i in num:
    sum = previousNum + i
    print('Current Number '+ str(i) + 'Previous Number ' + str(previousNum) + 'is ' + str(sum)) # <- This is the issue.
    previousNum=i


## video

````

# import the opencv library 
import cv2 
  
  
# define a video capture object 
vid = cv2.VideoCapture(0) 
  
while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
  
    # Display the resulting frame 
    cv2.imshow('frame', frame) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 




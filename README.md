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
![7622202030987_f306535d741c9148dc458acbbc887243_L_487](https://github.com/pavaniankam/pavani/assets/169125920/cb37a229-b7ca-4a0e-973a-29cbbbef1669)

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
5. It also saves the original images with drawn bounding boxes




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

![full_7622202030987_f306535d741c9148dc458acbbc887243_L_529](https://github.com/pavaniankam/pavani/assets/169125920/5efc7b12-d133-41f4-9bee-b02af6bbd480)
![0_7622202030987_f306535d741c9148dc458acbbc887243_L_488](https://github.com/pavaniankam/pavani/assets/169125920/8d5eb1b7-513e-474a-8b88-4ae943f144b3)

## Imagehistogram
```
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

![blue-cars](https://github.com/pavaniankam/pavani/assets/169125920/baa91c38-b5ea-473c-853b-83a52cd8a440)


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

![Figure_1](https://github.com/pavaniankam/pavani/assets/169125920/5272e70b-6280-470c-ad0a-044081045328)

## Iteration
```
num = list(range(10))
previousNum = 0
for i in num:
```
1.Creates a list containing numbers from 0 to 9.
2.Initializes a variable 'previousNum' to store the previous number in the iteration.
3.Iterates through each number in the 'num' list.
```
sum = previousNum + i
print('Current Number '+ str(i) + 'Previous Number ' + str(previousNum) + 'is ' + str(sum)) # <- This is the issue.
previousNum=i
```
1.Calculates the sum of the current number 'i'and the previous number 'previousNum'.

2.This line prints the current number (i), the previous number (previousNum), and their sum (sum).

3.Updates the previousNum variable for the next iteration.

4.statement will display the current number, the previous number, and their sum in a more readable format.
# Output
Current Number 0Previous Number 0is 0

Current Number 1Previous Number 0is 1

Current Number 2Previous Number 1is 3

Current Number 3Previous Number 2is 5

Current Number 4Previous Number 3is 7

Current Number 5Previous Number 4is 9

Current Number 6Previous Number 5is 11

Current Number 7Previous Number 6is 13

Current Number 8Previous Number 7is 15

Current Number 9Previous Number 8is 17

## Video
```
import cv2
```
1.Imports the OpenCV library
```
vid = cv2.VideoCapture(0)
```
1.Creates a VideoCapture object named vid which represents the video stream from the default camera (index 0).
```
while(True): 
      ret, frame = vid.read()
```
1.Reads a frame from the video stream.
2.ret is a boolean value indicating whether the frame was successfully read.
3.frame is the image frame.
```
 cv2.imshow('frame', frame)
```
1.cv2.imshow() is a function in OpenCV used to display images or videos.
```
      if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
```
1.cv2.waitKey() is a function that waits for a key event in a specified time duration.
```
vid.release() 
cv2.destroyAllWindows() 
```
1.Releases the VideoCapture object, releasing the camera.


https://github.com/pavaniankam/pavani/assets/169125920/d0ec0f95-8a5f-4cf5-b01c-e7104f6ecc74


2.Closes all OpenCV windows.
  



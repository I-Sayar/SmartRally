
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_equations(path):
    image = cv2.imread(path)

    #if not green make it white --> will work cause only the court itself is green and the lines of the court are already white so this will get rid of the useless background that could add extra lines which arent oart of the court
    for i in range(image.shape[0]):  
        for j in range(image.shape[1]):  
            blue = image[i, j, 0]
            green = image[i, j, 1]
            red = image[i, j, 2]
            
            if not (green > red and green > blue):
                image[i, j] = [255, 255, 255]  

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(gray_image, 190, 255, cv2.THRESH_BINARY) #threshold white or black so easier for canny and hough

    edges = cv2.Canny(binary_image, 100, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 90, 90, None, 10, 250)

    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        if abs(y2 - y1) < 0.5: #are y vaues close 
            horizontal_lines.append(line)
        
        elif abs(x2 - x1) < 280: 
            vertical_lines.append(line)

    #boundary lines
    top_line = None
    bottom_line = None
    left_line = None
    right_line = None
    highest_y = 10000000000  
    lowest_y = -1      
    leftmost_x = 10000000000 
    rightmost_x = -1    

    for line in horizontal_lines:
        x1, y1, x2, y2 = line[0]
        y_avg = (y1 + y2) / 2
        if y_avg < highest_y:
            highest_y = y_avg
            top_line = line
        if y_avg > lowest_y:
            lowest_y = y_avg
            bottom_line = line

    for line in vertical_lines:
        x1, y1, x2, y2 = line[0]
        
        x_avg = (x1 + x2) / 2
        if x_avg < leftmost_x:
            leftmost_x = x_avg
            left_line = line
        if x_avg > rightmost_x:
            rightmost_x = x_avg
            right_line = line
    equations = []
    if top_line is not None:
        x1, y1, x2, y2 = top_line[0]
        m = 0
        b = round((y1 + y2) / 2, 2)
        # print(f"top line equation: y = {b}")
        equations += [(float(m),float(b))]

    if bottom_line is not None:
        x1, y1, x2, y2 = bottom_line[0]
        m = 0
        b = round((y1 + y2) / 2, 2)
        # print(f"bottom line equation: y = {b}")
        equations += [(float(m),float(b))]

    if left_line is not None: #this is actually rightside
        x1, y1, x2, y2 = left_line[0]
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        # print(f"left line equation: y = {m}x + {b}")
        equations += [(float(m),float(b))]

    if right_line is not None: #this is left side
        x1, y1, x2, y2 = right_line[0]
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        # print(f"Right line equation: y = {m}x + {b}")
        equations += [(float(m),float(b))]


    #draw liens
    result_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    if top_line is not None:
        x1, y1, x2, y2 = top_line[0]
        cv2.line(result_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

    if bottom_line is not None:
        x1, y1, x2, y2 = bottom_line[0]
        cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    if left_line is not None:
        x1, y1, x2, y2 = left_line[0]
        cv2.line(result_image, (x1, y1), (x2, y2), (0, 0, 255), 3)

    if right_line is not None:
        x1, y1, x2, y2 = right_line[0]
        cv2.line(result_image, (x1, y1), (x2, y2), (255, 255, 0), 3)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.show()
    return equations

# print(get_equations("detect.png"))
#techanalcy only need very min and max x y cause the slight slant in the court sides will only make so much of a difference
#were only trying to detect court to get keypoints that are inside the court so it should be okay
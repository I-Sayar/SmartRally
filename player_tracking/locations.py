import cv2
from ultralytics import YOLO
from court_detection import get_equations
def get_centers(court_cords, num):
    _, top = court_cords[0]
    _, bottom = court_cords[1]
    right_m, right_b = court_cords[3]
    left_m, left_b = court_cords[2]
    tv = (((bottom - top)/(2))-120)/num
    # bv = (((bottom - top)/(2))+120)/num
    s = []
    for i in range(num):
        right = ((top+tv/2)+(i*tv) - right_b)/right_m
        left = ((top+tv/2)+(i*tv) - left_b)/left_m + 75
        horizontal = (right-left)/num
        for j in range(num):
            s += [(int(left+horizontal/2+(j*horizontal)),int(top+tv/2+(i*tv)))]
    # for i in range(num):
    #     right = ((top+tv*num+bv/2)+(i*bv) - right_b)/right_m
    #     left = ((top+tv*num+bv/2)+(i*bv) - left_b)/left_m + 75
    #     horizontal = (right-left)/num
    #     for j in range(num):
    #         s += [(int(left+horizontal/2+(j*horizontal)),int((top+tv*num+bv/2)+(i*bv)))]
    return s

def graph_shuttlecock_center(num,frame, centers,color):
    cv2.circle(frame, centers[num], radius=15, color=color, thickness=-1)  # red dots
    return frame

def graph_person(img,color):
    
    model = YOLO("yolov8m.pt")
    image = cv2.imread(img)
    equations = get_equations(image)
    results = model(image)
    boxes = results[0].boxes  

    for box in results[0].boxes:
        xmin, ymin, xmax, ymax = box.xyxy[0]  
        if xmin > equations[0][0] and xmax < equations[0][2] and ymin > equations[0][1] and ymax < equations[1][1]:
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)  
    return image

def get_captioned(frame, c1, c2, caption1, caption2): #make it so there is colors 
    height, width = frame.shape[:2]
    if c1: #means pose is correct  
        text = f"CORRECT: Best possible shot, {caption1}, is played. "
        text1 = "CORRECT: Best possible shot, "
        text2 = str(caption1)
        text3 = ", is played."
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,1, 2)[0]
        text_x = (width - text_size[0]) // 2
        text1_size = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX,1, 2)[0][0]
        text2_size = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX,1, 2)[0][0]
        cv2.putText(frame,text1,(text_x,height-50),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 0),cv2.LINE_AA)
        cv2.putText(frame,text2,(text_x + text1_size,height-50),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 100, 0),3,cv2.LINE_AA)
        cv2.putText(frame,text3,(text_x + text1_size + text2_size,height-50),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 0),2,cv2.LINE_AA)
    else: #incorect pose
        text = f"IMPROVEMENT: The best possible shot is {caption1[0]}, but {caption1[1]} was played instead."
        text1 = "IMPROVEMENT: The best possible shot is "
        text2 = str(caption1[0])
        text3 = ", but "
        text4 = str(caption1[1])
        text5 = " was played instead."
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,1, 2)[0]
        text_x = (width - text_size[0]) // 2
        text1_size = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX,1, 2)[0][0]
        text2_size = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX,1, 2)[0][0]
        text3_size = cv2.getTextSize(text3, cv2.FONT_HERSHEY_SIMPLEX,1, 2)[0][0]
        text4_size = cv2.getTextSize(text4, cv2.FONT_HERSHEY_SIMPLEX,1, 2)[0][0]
        cv2.putText(frame,text1,(text_x,height-50),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 0),2,cv2.LINE_AA)
        cv2.putText(frame,text2,(text_x + text1_size,height-50),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 100, 0),3,cv2.LINE_AA)
        cv2.putText(frame,text3,(text_x + text1_size + text2_size,height-50),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 0),2,cv2.LINE_AA)
        cv2.putText(frame,text4,(text_x+ text1_size+text2_size+text3_size,height-50),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 100),3,cv2.LINE_AA)
        cv2.putText(frame,text5,(text_x+ text1_size+text2_size+text3_size + text4_size,height-50),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 0),2,cv2.LINE_AA)
    if c2: #means direction is correct
        text = f"CORRECT: The best possible direction is Section {caption2} of the opponents court."
        text1 = "CORRECT: The best possible direction is Section "
        text2 = str(caption2)
        text3 = " of the opponents court."
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,1, 2)[0]
        text_x = (width - text_size[0]) // 2
        text1_size = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX,1, 2)[0][0]
        text2_size = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX,1, 2)[0][0]
        cv2.putText(frame,text1,(text_x,height-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 0),2,cv2.LINE_AA)
        cv2.putText(frame,text2,(text_x + text1_size,height-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 100, 0),3,cv2.LINE_AA)
        cv2.putText(frame,text3,(text_x + text1_size + text2_size,height-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 0),2,cv2.LINE_AA)
    else:   
        text = f"IMPROVMENT: Best possible direction is Section {caption2[0]}, but you played Section {caption2[1]}."
        text1 = "IMPROVMENT: Best possible direction is Section "
        text2 = str(caption2[0])
        text3 = ", but you played Section "
        text4 = str(caption2[1])
        text5 = "."
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,1, 2)[0]
        text_x = (width - text_size[0]) // 2
        text1_size = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX,1, 2)[0][0]
        text2_size = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX,1, 2)[0][0]
        text3_size = cv2.getTextSize(text3, cv2.FONT_HERSHEY_SIMPLEX,1, 2)[0][0]
        text4_size = cv2.getTextSize(text4, cv2.FONT_HERSHEY_SIMPLEX,1, 2)[0][0]
        cv2.putText(frame,text1,(text_x,height-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 0),2,cv2.LINE_AA)
        cv2.putText(frame,text2,(text_x + text1_size,height-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 100, 0),3,cv2.LINE_AA)
        cv2.putText(frame,text3,(text_x + text1_size + text2_size,height-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 0),2,cv2.LINE_AA)
        cv2.putText(frame,text4,(text_x+ text1_size+text2_size+text3_size,height-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 100),3,cv2.LINE_AA)
        cv2.putText(frame,text5,(text_x+ text1_size+text2_size+text3_size + text4_size,height-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 0),2,cv2.LINE_AA)
    return frame#cv2.addWeighted(frame, alpha, frame, 1 - alpha, 0)
 
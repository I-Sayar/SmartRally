from ultralytics import YOLO
import cv2
def get_keypoints(filename, equations):
    model = YOLO('yolo11m-pose.pt') 

    # court_coords = (400, 315-75, 1500, 900)  
    results = model(source=filename, show=False, conf=0.3, save=True) # + walrus <3
    keypoints = []

    for index, result in enumerate(results):
        # frame = result.orig_img  
        frame = []
        # x1, y1, x2, y2 = court_coords
        for i, pose in enumerate(result.keypoints):
            player = []
            #maybe switch to a csv file --> would be easier to read back in
            #f.write(f"Player {i}: ") #player " + i + " ") #move down, want only the actual players' data extraction
            # keypoints = pose.xy  いらん
            
            for index, keypoint in enumerate(p:=(pose.xy[0])):
                x, y = keypoint[0].item(), keypoint[1].item()

                if is_in(equations,(x,y)):
                    player += [(x,y)]
                    # f = open(f"{vid}_player{i}.txt", "a") #vid + "player" + str(i) + ".txt", "a") 
                    # f.write(f"({x},{y}), ")
                    # if index == len(p)-1:
                    #     f.write("\n")
                    # f.close()
                    # cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)  
            frame += [player]
        keypoints += [frame]
    return keypoints

def is_in(equations, point):
    _, top_b = equations[0]
    _, bottom_b = equations[1]
    right_m, right_b = equations[2]
    left_m, left_b = equations[3]
    x,y = point
    print(equations)
    if left_m == 0:
        input()
    if not(top_b-75 <= y <= bottom_b and (y-left_b)/left_m > x  and (y-right_b)/right_m < x):
        return False
    return True

def get_split(equations, num):
    _, top = equations[0]
    _, bottom = equations[1]
    right_m, right_b = equations[2]
    left_m, left_b = equations[3]
    vertical = (bottom - top)/(num*2) #3 for each court
    x = [] #gonna get 28
    y = [] #gonna get 7 diff values and 4 of each val --> 28 total
    for i in range(7):
        increment =((top+(i*vertical) - right_b)/right_m - (top+(i*vertical) - left_b)/left_m)/num
        for j in range(num+1):
            x += [top+(i*vertical)+increment*(j)]
            y += [top+(i*vertical)]
    eqs= []
    for i in range(len(x)-num-1):
        m = (y[i]-y[i+num+1])/(x[i]- x[i+num+1])
        b = (y[i]-m*x[i])
        eqs += [(m,b)]
        if i == 0 or i % num != 0: #skip the last one in the line
            eqs += [(0,y[i])]
            eqs += [(0,y[i+num+1])]
        else:
            eqs += []
    print(eqs)
    cords = []
    for i in range(0, len(eqs), 3):
        if i % num*3 != 0: #last one in line is never a left corner
            l = eqs[i]
            r = eqs[i+1]
            t = eqs[i+2]
            b = eqs[i+3]
            cords += [[t,b,r,l]]
    return cords

def get_locations(filename, equations):
    keypoints = get_keypoints(filename, equations)
    split = get_split(equations, 3)
    locations = []
    nonempty = []
    for k in keypoints:
        if len(k[0]) != 0:
            nonempty = k
    for i in range(len(nonempty)):
        for index, s in enumerate(split): 
            # x = (nonempty[i][19][0] + nonempty[i][20][0] + nonempty[i][21][0] + nonempty[i][22][0] + nonempty[i][23][0] + nonempty[i][24][0])/6 #average left and right foot 
            # y = (nonempty[i][19][1] + nonempty[i][20][1] + nonempty[i][21][1] + nonempty[i][22][1] + nonempty[i][23][1] + nonempty[i][24][1])/6
            x = (nonempty[i][-1][0] + nonempty[i][-2][0])/2 #left foot right foot
            y = (nonempty[i][-1][1] + nonempty[i][-2][1])/2
            if is_in(s,(x,y)): 
                if index < 9:
                    locations += [(1,9-index)]
                else:   
                    locations += [(0,index%9)]
                break
    return locations


# print(get_keypoints("match2.mp4"))

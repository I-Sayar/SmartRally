#current, hit_area, land_area, player_location_area, opponent_location_area, next

from ..player_tracking.skeletal_extraction import get_locations
from ..player_tracking.court_detection import get_equations
from nbm_predict import predict
from ..player_tracking.locations import graph_shuttlecock_center, graph_person, get_captioned, get_centers
from ..app.mobile import get_filname
from ..pose_estimation.pose_recognition import get_actual, get_splits
print("done")

filename = get_filname()#"match.mp4" #get this from the app upload


poses = ["Cut", "Block", "Smash", "Tap Smash", "Lift", "Defensive Clear", "Clear", "Short Flat Shot" ,# --> drive?
           "Flat Shot", "Rear-Court Flat Drive", "Drop Shot", "Push Shot", "Rush Shot",
            "Defensive Drive","Cross-Court Flight", "Short Serve", "Long Serve", "Unknown Serve"]


import cv2  
# vidObj = cv2.VideoCapture(filename) 

# count = 0
# success = 1

# while success: 
#     success, image = vidObj.read() 
#     cv2.imwrite(f"frames/frame{count}.png", image) 
#     count += 1
court_eqs = get_equations(f"frames/frame{int(132)}.png") #i dont wanna do the very first or last one cause it might be a black screen
sections = get_centers(court_eqs,4) #for birdies
print(sections)
print("nope")
videos = []#use pose recognition, every time a pose is recognized, save the video
outputs = []
actual = []#record poses played
#inputs = []
for v in videos:
    current = poses.index("Short Serve")#pose recognition here --> everytime is detects a pose, split the video there and send the information below

    hit_area = 1 #tracknet birdie recgonition here

    land_area = 2 #tracknet birdie recognition here

    location = get_locations(filename, court_eqs)#[(0,2), (1,6)]#skeletal_extraction.py
    print(location)
    for l in location:
        player, loc = l
        if player == 1:         
            player_location_area = loc #whoever is back doing the shots toward the guy in front
        else:
            opponent_location_area = loc
    #inputs += [[current, hit_area, land_area, player_location_area, opponent_location_area]]
    outputs += [[predict([current, hit_area, land_area, player_location_area, opponent_location_area]),15-player_location_area]] #nbm_alogorithm.py
#at the end of this we will have actual and predicited.

accuracy = []
for index, value in enumerate(actual):
    if value != outputs[index]:
        accuracy += [0]
    else:
        accuracy+=[1]



img = cv2.imread("frames/frame1.png")  # or from a video frame
height, width = img.shape[:2]
# for n in range(int(len(sections))):
#     cv2.circle(img, sections[n], radius=10, color=(0,0,0), thickness=-1)  # red dots
#     cv2.imwrite("8_sections.png", img)
#     input()
# cv2.imshow(img)
# input()
out = cv2.VideoWriter(f'vid.mp4', cv2.VideoWriter_fourcc(*'XVID'), 30,(width,height))
out.write(img)
farthest = {0: 15, 1: 15, 2: 12, 3: 12, 4: 15, 5: 15, 6: 12, 7: 12, 8: 3, 9: 3, 10: 0, 11: 0, 12: 3, 13: 3, 14: 0, 15: 0}
actual = get_actual(filename)#[["Lift", 2], ["Clear", 9], ["Flat Shot", 10],["Lift", 13], ["Defensive Drive", 13], ["Defensive Clear", 15], ["Clear", 14], ["Defensive Drive", 15], ["Dropshot", 1], ["Cross-Court Flight", 15], ["Cut", 5], ["Tap Smash", 9], ["Defensive Drive", 14]] #[["Cross-Court Flight", 12], ["Tap Smash", 11], ["Flat Shot", 8], ["Drop Shot", 0], ["Tap Smash", 12], ["Drop Shot", 0], ["Cross-Court Flight", 11], ["Short Flat Shot", 13], ["Lift", 13], ["Rush Shot", 8], ["Rush Shot", -1]]

splits = get_splits()# [] #will contrain the frame numbers of where each pose section ends. 
count = 0
success = 1
vidObj = cv2.VideoCapture("match3.mp4") 

while success: 
    success, image = vidObj.read() 
    cv2.imwrite(f"frames/frame{count}.png", image) 
    count += 1
for index in range(1,splits[0]):
    out.write(cv2.imread(f"frames/frame{count}.png"))
    count +=1

for index, a in enumerate(actual):
    #out = cv2.VideoWriter(f'videos/{index}.mp4', cv2.VideoWriter_fourcc(*'XVID'), 30,(width,height))
    while count < splits[index]:
        actual_shot, acutal_direction = a
        predicted_shot, predicted_direction = outputs[index]
        caption = ""
        if actual_shot == str(predicted_shot):
            frame = graph_person(f"frames/frame{count}.png", (0,255,0))
            caption += f"CORRECT: Best possible shot, {actual_shot}, is played.\n"
        else:
            frame = graph_person(f"frames/frame{count}.png", (0,0,255))
            caption += f"IMPROVEMENT: The best possible shot is {predicted_shot}, but {actual_shot} was played instead.\n"
        # if acutal_direction < len(sections)/2:
        #     ad = 15-acutal_direction
        # if predicted_direction < len(sections)/2:
        #     pd = 15-predicted_direction
        if acutal_direction != -1: # if not end game
            if acutal_direction == predicted_direction:
                frame = graph_shuttlecock_center(acutal_direction,frame, sections,(0,255,0))
                caption += f"CORRECT: The best possible direction is section {acutal_direction} of the opponents court. "
            else:
                frame = graph_shuttlecock_center(acutal_direction,frame, sections,(0,0,255))
                frame = graph_shuttlecock_center(predicted_direction,frame, sections,(0,255,0))
                caption += f"IMPROVMENT: Best possible direction is section {predicted_direction}, but you played {acutal_direction}."
        frame = get_captioned(frame, caption)
        count+=1
        out.write(frame)
    




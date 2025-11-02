import csv

english = ["Cut", "Block", "Smash", "Tap Smash", "Lift", "Defensive Clear", "Clear", "Short Flat Shot" ,# --> drive?
           "Flat Shot", "Rear-Court Flat Drive", "Drop Shot", "Push Shot", "Rush Shot",
            "Defensive Drive","Cross-Court Flight", "Short Serve", "Long Serve", "Unknown Serve"] #okay we have this here but since doing a NN, i wanna stick to numbers so i will use the indicies instead of the actual english name and when grabbing the classifications back we could possibly turn it from CN to ENG
chinese = ["放小球", "擋小球", "殺球", "點扣", "挑球", "防守回挑", "長球", "平球", "小平球", "後場抽平球", "切球",
             "推球", "撲球", "防守回抽", "勾球", "發短球", "發長球", "未知球種"]
#"Transitional Slice" "過渡切球" got rid of for now --> never played in any of the matches
# matches = dict()
# nexts = []
# locations =[]
# for i in range(104):
#     with open(f"dataset/set{i}.csv", "r", encoding="utf-8") as file:
#         csv_reader = csv.reader(file)
#         for row in csv_reader:
#             if row[8] != "type" and row[8] != "過度切球" and row[8] != "" and row[16] != "":
#                 nexts += [[chinese.index(row[8]), int(row[16])]]
# nexts = nexts[1:]
# player = []
# for i in range(104):
#     with open(f"dataset/set{i}.csv", "r", encoding="utf-8") as file:
#         csv_reader = csv.reader(file)
#         rows = []
#         location = [4,]
#         for row in csv_reader:
#             new_row = []
#             add = True
#             if row[6] != "" and row[12] != "" and row[16] != "" and row[23] != "" and row[26] != "" and row[8] != "":
#                 for ind, rw in enumerate(row):
#                     if rw == "A":
#                         row[ind] = 0
#                     elif rw == "B":
#                         row[ind] = 1
#                 for index, word in enumerate(chinese): 
#                     if row[8] == word:
#                         new_row += [index, row[12],row[16],row[23], row[26], nexts[0][0], nexts[0][1]] #previous, hit_area, land_area, player_location_area, opponent_location_area, shot
#                         nexts = nexts[1:]
#                         player += [row[6]]
#                 rows += [new_row]
#     matches[i] = rows

# with open('dataset/datasettt.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile,lineterminator='\n')
#     for m in matches:
#         for row in matches[m]:
#             writer.writerow(row)

count = dict()
best_2 = dict()
likely_next = dict()
likely_next_total = dict()
scoring = dict()
scoring_total = 0
for sho in english:
    likely_next[sho]= dict()
    likely_next_total[sho] = 0
    scoring[sho] = 0
    count[sho] = 0
    for s in english:
        likely_next[sho][s] = 0
# with open("matches.txt","w") as file:   
#     len_previous = 0     
#     for m in matches:
#         # shot = matches[m][1]
#         # player = matches[m][0]
#         file.write(f"Match {m+1}: \n")
#         for i in range(len_previous, len_previous:=(len_previous+len(matches))):
#             file.write(f"Player {player[i]} played the move {shot[i]}\n")
#             if i < len(shot)-1:
#                 likely_next[shot[i]][shot[i+1]] +=1
#                 likely_next_total[shot[i]] +=1
#                 if player[i] == player[i+1]:
#                     file.write(f"Player {player[i]} scored a point.\n")
#                     scoring[shot[i]] +=1
#                     scoring_total +=1 
#             count[shot[i]] +=1
#         scoring[shot[len(shot)-1]] +=1
#         file.write(f"Player {player[len(shot)-1]} won the match.\n\n")

most_played = 0
most_key =""
least_played = 100000000
least_key =""
count = dict(sorted(count.items(), key=lambda item: item[1], reverse=True))

#have a good stats files, dont wanna mess it up rn
with open("statistics.txt", "w") as file:
    file.write("Likely next move :\n")
    for sho in likely_next:
        file.write(f"{sho} :\n")
        for s in dict(sorted(likely_next[sho].items(), key=lambda item: item[1], reverse=True)):
            if likely_next_total[sho] != 0:
                if (normalize:=(likely_next[sho][s]/likely_next_total[sho])) == 0.0:
                    file.write(f"   Dont play {s} after {sho}\n")
                else:
                    file.write(f"   {s} : {normalize}\n")
    file.write(f"\nMost common move following any previous move: Cut\n")
    file.write("\nBest move to score a point (/total score):\n")
    for num in dict(sorted(scoring.items(), key=lambda item: item[1], reverse=True)):
        if (count[num] != 0):
            best_2[num] =scoring[num]/count[num]
        file.write(f"   {num} : {scoring[num]/scoring_total}\n")
    file.write("\nBest move to score a point (/# of times move was played):\n")
    for num in dict(sorted(best_2.items(), key=lambda item: item[1], reverse=True)):
        file.write(f"   {num} : {best_2[num]}\n")
    total_moves = sum(count.values())
    file.write(f"\nMost to least likely to be played: (/total moves ({total_moves}))\n")
    for val in count:
        file.write(f"   {val} {count[val]/total_moves}\n" )

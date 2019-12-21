import os
import re

folder = "D:/Nike/music/"
arr = os.listdir(folder)

for a in arr:
    x = re.search("\([1-9]", a)
    
    if x != None:
        os.remove(folder + a)
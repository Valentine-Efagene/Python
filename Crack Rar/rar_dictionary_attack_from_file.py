import io
import os
import subprocess

devnull = open(os.devnull, 'w')

# Function to find permutations of a given string 
from itertools import permutations

#keys = ["valentyne", "password", "test"]
print("Running...")

with open('C:\\Users\\valentyne\\Desktop\\dictionary.txt', 'r') as fp:
    for key in fp:
        key = key.replace(" ", "").replace("\n", "")
        print(key)
        password = "-p" + key
        error = subprocess.call( ['unrar', 'x', password, 'C:\\Users\\valentyne\\Desktop\\test.rar'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL )

        if (error == 0):
            print(key)
            exit(0)
        else:
            #os.rmdir("C:\\Users\\valentyne\\Desktop\\test") # If folder
            continue
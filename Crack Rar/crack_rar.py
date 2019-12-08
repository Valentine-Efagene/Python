import os
import subprocess

devnull = open(os.devnull, 'w')

# Function to find permutations of a given string 
from itertools import permutations 

def allPermutations(str):
	return permutations(str)

str = 'aec'
#str = 'sapsword'
keys = list(allPermutations(str))

#keys = ["valentyne", "password", "test"]
print("Running...")

for key in keys:
    password = ''.join(key)
    print(password)
    s = "-p" + password
    error = subprocess.call( ['unrar', 'x', s, 'C:\\Users\\valentyne\\Desktop\\test.rar'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL )

    if (error == 0):
        print(password)
        exit(0)
    else:
        #os.rmdir("C:\\Users\\valentyne\\Desktop\\test") # If folder
        continue
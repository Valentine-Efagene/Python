from os import path

if (not path.exists('model.h5')):
    print("There is no model to load!")
    exit(0)
    
from keras.models import load_model
import numpy as np

model = load_model('model.h5')

if (not path.exists('model.h5')):
    print("There is no model to load!")
    exit(0)

model = load_model('model.h5')

plans_MB = np.array([1500, 2000, 3500, 6500, 11000, 25000]);
usage_s = ""
print("Enter the letter \'e\' to exit\n");

while True:
    if  usage_s == "e":
        exit(0)

    usage_s = input("Enter Usage: ")
    usage = float(usage_s)
    usage /= 2500
    pred = model.predict([usage])
    print("You should use the", plans_MB[pred.argmax()], "MB data plan.\n\n")
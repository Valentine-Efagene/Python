from os import walk
import os
from  Tkinter import Entry, StringVar, Button, INSERT, Tk, Label
import Tkconstants, tkFileDialog
import Tkinter as tk
import pickle
import tkMessageBox

def showDirectory():
    directory = tkFileDialog.askopenfilename()
    folderPathText.delete(0, 'end')
    folderPathText.insert(INSERT, directory)

def save():
    try:
        cellNumber = int(cellNumberText.get())
        p_out = open(folderPathText.get(), 'wb')
        pickle.dump(cellNumber, p_out)
        p_out.close()   
    except ValueError:
        tkMessageBox.showerror('ERROR', 'Please enter an integer.')

def view():
    try:
        if os.path.exists(folderPathText.get()):
            p_in = open(folderPathText.get(), 'r')
            cellNumber = pickle.load(p_in)
            p_in.close()
            currentCellNumberText.delete(0, 'end')
            currentCellNumberText.insert(INSERT, cellNumber)
        else:
            tkMessageBox.showerror('ERROR', 'No file!')    
    except ValueError:
        tkMessageBox.showerror('ERROR', 'Please enter an integer.')


root = Tk()
root.geometry('500x100')
root.resizable(0, 0)

folderPathLabel = Label(master = root, text = 'Files Directory')

folderPathText = Entry(master = root, width = 50)

getDirBtn = Button(root, text="...", command=showDirectory, width = 3, padx = 10)
cellNumberLabel = Label(master = root, text = 'Cell Number')
currentCellNumberLabel = Label(master = root, text = 'Current Cell Number')

cellNumberText = Entry(master = root, width = 50)
currentCellNumberText = Entry(master = root, width = 50)

saveBtn = Button(root, text="save", command=save, width = 3, padx = 10)
currentCellNumberBtn = Button(root, text="Current", command=view, width = 3, padx = 10)

folderPathLabel.grid(row = 0, column = 0)
folderPathText.grid(row = 0, column = 1)
getDirBtn.grid(row = 0, column = 2)
cellNumberLabel.grid(row = 1, column = 0)
saveBtn.grid(row = 1, column = 2)
cellNumberText.grid(row = 1, column = 1)
currentCellNumberLabel.grid(row = 2, column = 0)
currentCellNumberText.grid(row = 2, column = 1)
currentCellNumberBtn.grid(row = 2, column = 2)
root.mainloop()
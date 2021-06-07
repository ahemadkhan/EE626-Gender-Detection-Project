# Done by Group 20 , EE626

import os
from tkinter import *

def detect():
    os.system("detect_gender_webcam.py")



def train():
    os.system("train.py")


window = Tk()
window.title("Gender Classification based on face detection")
window.geometry('800x600')
topFrame = Frame(window)
topFrame.pack(side=TOP)
bottomFrame = Frame(window)
bottomFrame.pack(side=BOTTOM)

train_button = Button(topFrame, text='Train Model', command=train, font="Helvetica", fg="black", bg="yellow")
start_button = Button(bottomFrame, text='Detect from webcam', command=detect, font="Helvetica", fg="black", bg="green")


train_button.pack()
start_button.pack()


window.mainloop()

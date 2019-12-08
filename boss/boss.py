import pyautogui as p
import keyboard as k

while True:
    if k.is_pressed('shift'):
        p.press('enter')
    else:
        pass
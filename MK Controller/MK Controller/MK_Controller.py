import pyautogui as p
import keyboard as k

STEP = 5
width, height = p.size()
x, y = p.position()

try:
    while True:
        while k.is_pressed('up'):
            y -= STEP
            p.moveTo(None, y)
        if k.is_pressed('down'):
            y += STEP
            p.moveTo(None, y)
        elif k.is_pressed('right'):
            x += STEP
            p.moveTo(x, None)
        elif k.is_pressed('left'):
            x -= STEP
            p.moveTo(x, None)
except KeyboardInterrupt:
    pass
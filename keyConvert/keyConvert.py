from pynput.keyboard import Key, Listener
import pyautogui as p

def on_press(key):
    if key == Key.shift_r:
        p.press('enter')

def on_release(key):
    return

    if key == key.esc:
        return False

with Listener(on_press = on_press, on_release=on_release) as listener:
    listener.join()
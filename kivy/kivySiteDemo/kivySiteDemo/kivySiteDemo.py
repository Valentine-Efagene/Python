from kivy.app import App
from kivy.uix.button import Button
import pygame

class TestApp(App):
    def build(self):
        return Button(text='Hello World')

TestApp().run()


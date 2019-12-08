import pyttsx3
#from pyttsx import voice

engine = pyttsx3.init();
voices = engine.getProperty('voices')
engine.setProperty('gender', voices[2].id) # I only have 0-1 which are essntially the same
engine.say("Hello World")
engine.runAndWait()
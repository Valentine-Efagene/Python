from Tkinter import *

class App( Frame ):
    def __init__( self, master = None ):
        Frame.__init__( self, master )
        self.pack()

        self.entrythingy = Entry()
        self.button = Button(self, text = 'button')
        self.entrythingy.pack()
        self.button.pack()

        #variable
        self.contents = StringVar()
        self.contents.set('this is a variable')
        self.entrythingy['textvariable'] = self.contents
        self.entrythingy.bind('<Key-Return>', self.print_contents)

        self.button.bind("<Enter>", self.turnRed)

    def print_contents(self, event):
        print 'hi. contents of entry is now ---->', self.contents.get()

    def turnRed(self, event):
        event.widget["activebackground"] = "red"

app = App()
app.mainloop()
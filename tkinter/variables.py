from Tkinter import *

class App( Frame ):
    def __init__( self, master = None ):
        Frame.__init__( self, master )
        self.pack()

        self.entrythingy = Entry()
        self.entrythingy.pack()

        #variable
        self.contents = StringVar()
        self.contents.set('this is a variable')
        self.entrythingy['textvariable'] = self.contents
        self.entrythingy.bind('<Key-Return>', self.print_contents)

    def print_contents(self, event):
        print 'hi. contents of entry is now ---->', self.contents.get()

app = App()
app.mainloop()
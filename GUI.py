from tkinter import filedialog
from tkinter import *
import os


class Window:
    def __init__(self, master):
        self.root = master
        Label(self.root, text="Load File:").grid(row=1, column=0)
        self.fileEntry = Entry(self.root)
        self.fileEntry.grid(row=1, column=1)
        # Buttons
        self.cbutton = Button(self.root, text="OK", command=self.process_csv)
        self.cbutton.grid(row=15, column=3, sticky=W + E)
        self.bbutton = Button(self.root, text="browse", command=self.browse)
        self.bbutton.grid(row=1, column=3)

    def browse(self):
        self.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                                   filetypes=(("text files", "*.txt"), ("all files", "*.*")))
        self.fileEntry.insert(0, os.path.basename(self.filename))

    def process_csv(self):
        pass


root = Tk()
window = Window(root)
root.mainloop()

import tkinter as tk
master = tk.Tk()

master.geometry("300x300")
master.attributes('-type', 'dialog')

whatever_you_do = "Whatever you do will be insignificant, but it is very important that you do it.\n(Mahatma Gandhi)"

button = tk.Button(master, text="Butt")
button.pack(side=tk.LEFT)
msg = tk.Message(master, text = whatever_you_do)
msg.config(bg='lightgreen', font=('times', 24, 'italic'))
msg.pack(side=tk.LEFT)

tk.mainloop()

import tkinter as tk

fields = ['First X Column Name', 'Last X Column Name', 'First Y Column name','Last Y Column Name']
entries = []

things=[]

def fetch(_=None):
    # things=[]
    # file1 = open(r"./config.txt","w+") 
    for ndex, entry in enumerate(entries):
        print('{}: {}'.format(fields[ndex], entry.get()))
        things.append(entry.get())
    return things

def getCreds(self):
    root=tk.Tk()
    root.config(background='gray')
    for ndex, field in enumerate(fields):
        tk.Label(root, width=20, text=field, anchor='w').grid(row=ndex, column=0, sticky='ew')
        entries.append(tk.Entry(root))
        entries[-1].grid(row=ndex, column=1, sticky='ew')


    tk.Button(root, text='Submit', command=fetch).grid(row=len(fields)+1, column=1, sticky='ew')
    tk.Button(root, text='Quit', command=root.quit()).grid(row=len(fields)+2, column=1, sticky='ew')
    root.mainloop()
    return things

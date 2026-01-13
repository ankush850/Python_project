from PyPDF2 import PdfMerger, PdfReader
import tkinter as tk
from tkinter import filedialog
import os

tk.Tk().withdraw()

def list():
    for line in [
        "1. merge()",
        "2. sub_pdf()",
        "3. delete_pages_by_range()",
        "4. insert_at_position()",
        "5. merge_pages()",
        "6. delete_pages()",
        " - to get function description, use __doc__. For example, merge().__doc__"
    ]:
        print(line)

# Merge multiple pdf files
def merge():
    '''
    This takes a list of files and merges them into a new pdf file in the given order.
    ''' #merge.__doc__

    files = []

    while True:
        file = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file == "":
            break
        files.append(file)
    
    if len(files):
        merger = PdfMerger()
        for pdf in files:
            merger.append(pdf)
        save_and_close(merger)


# Get subset of a pdf
def sub_pdf():          
    '''
    From the input pdf, this function cuts the pages in given range and provides them as a separate pdf.
    '''#sub_pdf.__doc__

    file = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])

    if file:
        print("enter starting and ending pages to get..")
        start, end = map(int,input().split())

        merger = PdfMerger()
        merger.append(file,pages=(start-1,end))
        save_and_close(merger)


# Cut the appendix
def delete_pages_by_range():
    '''
    From the input pdf, this function deletes the pages in given range and provides a separate pdf.
    '''#delete_pages.__doc__
    file = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])

    if file:
        print("enter starting and ending pages to delete: ")
        start, end = map(int,input().split())
        reader = PdfReader(file)

        merger = PdfMerger()
        merger.append(file,pages=(0,start - 1))
        merger.append(file,pages=(end,len(reader.pages)))
        save_and_close(merger)


# Puzzle-in at right place
def insert_at_position():
    '''
    Two pdf files are taken where a new file is created as an altered version of parent file, alter refers to inserting of child file at given position in parent file.
    '''#push_insert.__doc__
    parent = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")], title="Select parent pdf")

    if parent:
        child = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")], title="Select child pdf")

        if child:
            position = int(input('Enter position to insert: '))

            merger = PdfMerger()
            merger.append(parent)
            merger.merge(position = position - 1,fileobj = child)
            save_and_close(merger)


# Putting the pieces together
def merge_pages():
    '''
    In the given file, the pages are selected and merged with respect to the given page numbers to form a new pdf.
    Note: The new pdf will have pages in order as they are given in the input and if a page numbers repeats, the corresponding page will also be repeated in the new file.
    '''#select_and_merge.__doc__

    file = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])

    if file:
        print("Enter pages to select..")
        pages = list(map(int,input().split()))

        merger = PdfMerger()
        for page in pages:
            merger.append(file, pages=(page-1,page))
        save_and_close(merger)


# Cutting the thorns from rose stem
def delete_pages():
    '''
    In the given file, the given page numbers are removed from input file and creates a new file.
    '''#select_and_remove.__doc__

    file = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])

    if file:
        pages = set(map(int, input( "Enter pages to remove.." ).split()))
        pages = sorted(list(pages))

        merger = PdfMerger()
        prev = 0
        for page in pages:
            merger.append(file, pages=(prev, page-1))
            prev = page
        save_and_close(merger)


# Saving the updated pdf file
def save_and_close(merger):
    print('save the merged pdf..')

    # Create a file dialog to save new pdf file
    while True:
        file_dialog = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")], title='Save as')
        if file_dialog:
            break
        print("Invalid input\nTry again..")

    merger.write(file_dialog)
    merger.close()

    print(f"File saved as '{file_dialog}'")
    # Modified to open the PDF in a larger/more prominent way (OS-dependent)
    # On Windows, this uses the default app and may open maximized if configured
    # For cross-platform, consider using webbrowser or subprocess for more control
    import subprocess
    try:
        # Attempt to open with default app (often maximized on some systems)
        subprocess.run(['start', file_dialog], shell=True, check=True)  # Windows-specific
    except:
        # Fallback for other OS or if 'start' fails
        os.startfile(file_dialog) if hasattr(os, 'startfile') else os.system(f'xdg-open "{file_dialog}"')  # Linux/Mac fallback

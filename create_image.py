from tkinter import *
import image_data as data

root = Tk()
canvas = Canvas(root, width=28*10, height=28*10)
canvas.pack()

def draw(index):
    draw_pixelset(data.test_data[index])

def draw_pixelset(pixelset):
    x = 0
    y = 0
    for pixel in pixelset:
        canvas.create_rectangle((x, y, x+10, y+10), fill=("#%02x%02x%02x" % (pixel, pixel, pixel)))
        x += 10
        if (x >= 280):
            x = 0
            y += 10

def run():
    index = 0
    while True:
        draw(index)
        index += 1
        try :
            root.update()
            root.update_idletasks()
        except Exception as e:
            print(str(e))
            quit()

run()
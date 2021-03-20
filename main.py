from network import Network
import tkinter as tk
from tkinter import messagebox
import numpy as np

layers = [2, 4, 5, 3, 1]
size = 500
steps = int(size / 20)
window = tk.Tk()
canvas = tk.Canvas(window, width=size, height=size, bg="#fff")


def clearData():
    global data
    data = np.empty((0, 3), float)
    canvas.delete("points")


network = Network(layers)
clearData()


def train():
    network.train(data)
    drawText()


def translate(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return rightMin + (valueScaled * rightSpan)


def f(a, b):
    a /= steps
    b /= steps
    res = network.calculate([a, b])[0]
    green = int(translate(max(res, 0.5), 0.5, 1, 0, 255))
    blue = int(translate(min(res, 0.5), 0.5, 0, 0, 255))
    return f"#{255 - green - blue:02x}{int(255 - blue):02x}{int(255 - green):02x}"

def draw():
    canvas.delete("all")
    for y in range(steps):
        for x in range(steps):
            color = f(x, y)
            canvas.create_rectangle(size / steps * x, size / steps * y, size / steps * x + size / steps, size / steps * y + size / steps,
                                    fill=color, outline=color)
    drawData()
    drawText()
    window.update_idletasks()
    window.after(1000, draw)


def drawText():
    canvas.delete("text")
    canvas.create_text(2, 2, anchor="nw", tags="text", text=f"Поколение: {network.generations}\n"
                                                            f"Ошибка: {network.error}")


def drawData(last=False):
    offset = 4
    #TODO: refactor

    if len(data) > 0:
        if last:
            x = data[-1][0] * size
            y = data[-1][1] * size
            if data[-1][-1] == 0:
                color = "#22F"
            else:
                color = "#2F2"
            canvas.create_oval(x - offset, y - offset, x + offset, y + offset, outline="#000", fill=color, tags="points", width=2)
        else:
            for dataset in data * [size, size, 1]:
                if dataset[-1] == 0:
                    color = "#22F"
                else:
                    color = "#2F2"
                canvas.create_oval(dataset[0] - offset, dataset[1] - offset, dataset[0] + offset, dataset[1] + offset,
                                   outline="#000", fill=color, tags="points", width=2)


def recreate():
    global network
    del(network)
    network = Network(layers)
    draw()

def leftMButton(event=None):
    global data
    global leftHoldID
    leftHoldID = window.after(125, leftMButton)
    if event:
        x = event.x
        y = event.y
    else:
        x = window.winfo_pointerx() - window.winfo_x() - 193
        y = window.winfo_pointery() - window.winfo_y() - 27
    data = np.append(data, [[x / size, y / size, 1]], axis=0)
    drawData(True)

def leftRelease(event):
    global leftHoldID
    window.after_cancel(leftHoldID)

def rightMButton(event=None):
    global data
    global rightHoldID
    rightHoldID = window.after(125, rightMButton)
    if event:
        x = event.x
        y = event.y
    else:
        x = window.winfo_pointerx() - window.winfo_x() - 193
        y = window.winfo_pointery() - window.winfo_y() - 27
    data = np.append(data, [[x / size, y / size, 0]], axis=0)
    drawData(True)

def rightRelease(event):
    global rightHoldID
    window.after_cancel(rightHoldID)

def about():
    try:
        with open("О работе.txt", "r", encoding="utf-8") as about_file:
            text = about_file.read()
    except FileNotFoundError:
        text = "#TODO: make about text"
    messagebox.showinfo("О программе", text)


def howto():
    text = "Для начала нужно добавить входные данные.\n" \
           "Левая кнопка мыши создаёт зелёную точку\n" \
           "Правая кнопка мыши создаёт синюю точку\n\n" \
           "Когда необходимые точки заданы - следует нажать на кнопку " \
           "\"Тренировать нейронную сеть\". Каждое нажатие симулирует " \
           "100 поколений обучения нейронной сети.\n\nДля удаления входных " \
           "данных, и создания новой сети есть соответствующие кнопки"
    messagebox.showinfo("Инструкция", text)


drawButton = tk.Button(window, text="Визуализировать", command=draw)
trainButton = tk.Button(window, text="Тренировать нейронную сеть", command=train)
recreateButton = tk.Button(window, text="Пересоздать нейронную сеть", command=recreate)
clearDataButton = tk.Button(window, text="Очистить входные данные", command=clearData)
aboutButton = tk.Button(window, text="О программе", command=about)
howtoButton = tk.Button(window, text="Инструкция по использованию", command=howto)

trainButton.grid(column=0, row=0, stick="NEWS")
recreateButton.grid(column=0, row=1, stick="NEWS")
clearDataButton.grid(column=0, row=2, stick="NEWS")
aboutButton.grid(column=0, row=3, stick="NEWS")
howtoButton.grid(column=0, row=4, stick="NEWS")
canvas.grid(column=1, row=0, rowspan=5, columnspan=5)
window.title("Нейронная сеть")
draw()
canvas.bind("<Button-1>", leftMButton)
canvas.bind("<ButtonRelease-1>", leftRelease)
canvas.bind("<Button-3>", rightMButton)
canvas.bind("<ButtonRelease-3>", rightRelease)
tk.mainloop()

from ultralytics import YOLO
import tkinter as tk
import cv2
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk
from tkinter import messagebox
from PIL import Image
import numpy as np
np.set_printoptions(suppress=True)
top = tk.Tk()
top.geometry('1200x800')
top.title('Counting humans in image using Machine Learning')
img= PhotoImage(file='bgrt.png', master=top)
img_label= Label(top,image=img)
img_label.place(x=0, y=0)

def classify(file_path):
    model = YOLO('best.pt')
    image_path = file_path
    image = cv2.imread(image_path)
    results = model(image)
    detections = results[0].boxes
    class_counts = {}

    for detection in detections:
        class_id = int(detection.cls)  # Get class id
        if class_id in class_counts:
            class_counts[class_id] += 1
        else:
            class_counts[class_id] = 1
    totalcount = 0
    # Print the class counts
    for class_id, count in class_counts.items():
        print(f"Class {class_id}: {count} instances")
        totalcount = count
    strcount = str(totalcount)
    messagebox.showinfo("Message", "Total Count:" + strcount)
    annotated_image = results[0].plot()
    cv2.imwrite('annotated_image.jpg', annotated_image)

    uploaded = Image.open("annotated_image.jpg")
    uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
    im = ImageTk.PhotoImage(uploaded)
    resultimg.configure(image=im)
    resultimg.image = im
    label.configure(text='')

def webcamfun():
    model = YOLO('best.pt')
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(mode="predict", model="best.pt", conf=0.95, classes=[0], source=frame)
        detections = results[0].boxes
        class_counts = {}
        for detection in detections:
            class_id = int(detection.cls)
            if class_id in class_counts:
                class_counts[class_id] += 1
            else:
                class_counts[class_id] = 1
        totalcount = 0
        for class_id, count in class_counts.items():
            print(f"Class {class_id}: {count} instances")
            totalcount = count
        strcount = str(totalcount)
        annotated_image = results[0].plot()
        cv2.putText(frame, f'Person Count: {strcount}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        annotated_image = results[0].plot()
        cv2.imshow('Webcam Feed', annotated_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image",command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.60, rely=0.80)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25),(top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass



label = Label(top)
label = Label(top, background='white', font=('Times New Roman', 15, 'bold'))

sign_image = Label(top)
sign_image.place(relx=0.10, rely=0.26)

resultimg=Label(top)
resultimg.place(relx=0.50, rely=0.26)

upload = Button(top, text="Upload an image",  padx=10, pady=5)
webcam = Button(top, text="webcam", command=webcamfun, padx=21, pady=5)
webcam.pack(side=BOTTOM,padx=10,pady=10)

upload.configure(background='#364156', foreground='white', command=upload_image, font=('Times New Roman', 10, 'bold'))

upload.pack(side=BOTTOM, pady=50)

label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Counting humans in image using Machine Learning", pady=20, font=('Times New Roman', 20, 'bold'))

heading.configure(background='#66FFFF', foreground='#660033')


heading.pack()
top.mainloop()

# Image classification model to classify images of benign or malignant skin cancer
# Benign amount = 4999
# Malignant amount = 9604

import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
import image
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
import customtkinter
import tkinter as tk
from PIL import Image, ImageTk


def test_against_model(file):
    # load model
    model = keras.models.load_model('model.h5')

    # run file through model
    img_size = 224
    img = image.load_img(file, target_size=(img_size, img_size))
    img = image.img_to_array(img)
    img = img/255.0

    img = np.expand_dims(img, axis=0)
    result = model.predict(img)

    if result >= 0.1:
        fileresult = "Malignant"
    else:
        fileresult = "Benign"

    # create label for result
    label = tk.Label(app, text=fileresult, font=("Arial", 20), bg="#111111", fg="white", width=10)
    label.place(relx=0.5, rely=0.7, anchor=customtkinter.CENTER)

    # change background colour of label
    if fileresult == "Malignant":
        label.config(bg="red")
    else:
        label.config(bg="green")


def display_img(file):
    my_image = customtkinter.CTkImage(light_image=Image.open(file),
                                      dark_image=Image.open(file),
                                      size=(250, 150))

    my_label = customtkinter.CTkLabel(app, text="", image=my_image)
    my_label.pack()

    my_label.place(relx=0.5, rely=0.35, anchor="center", y=10)


def upload_file():
    file = customtkinter.filedialog.askopenfilename()

    display_img(file)
    test_against_model(file)


def train_model():
    # Load the images
    img_size = 224
    datadir = 'dataset/train'
    categories = ['benign', 'malignant']
    data = []

    for category in categories:
        path = os.path.join(datadir, category)
        label = categories.index(category)
        for img in os.listdir(path):
            img_array = image.load_img(os.path.join(path, img), target_size=(img_size, img_size))
            img_array = image.img_to_array(img_array)
            img_array = img_array/255.0
            data.append([img_array, label])

    # Split the data into features and labels
    x = []
    y = []
    for features, label in data:
        x.append(features)
        y.append(label)

    # Convert the data into numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Create the model
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    # save model
    model.save('model.h5')


customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("green")  # Themes: blue (default), dark-blue, green

app = customtkinter.CTk()  # create CTk window like you do with the Tk window
app.geometry("350x400")
app.title("Skin Cancer Image Classifier")
app.config(bg="#111111")

# Use CTkButton instead of tkinter Button
button = customtkinter.CTkButton(master=app, text="Test file", command=upload_file, height=30, width=250,
                                 text_color="black")
button.place(relx=0.5, rely=0.9, anchor=customtkinter.CENTER)

# Title at the top of gui
label = tk.Label(app, text="Skin Cancer Image Classifier", font=("Arial", 20), bg="#111111", fg="white")
label.place(relx=0.5, rely=0.1, anchor=customtkinter.CENTER)

# button = customtkinter.CTkButton(master=app, text="Train model", command=train_model)
# button.place(relx=0.5, rely=0.4, anchor=customtkinter.CENTER)

app.mainloop()
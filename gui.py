from tkinter import Tk, Label, Button, filedialog, StringVar, Entry, messagebox
import os
import numpy as np
from keras.preprocessing import image
from keras import layers
from keras import models
from keras.models import load_model
from keras.layers import Dense, Flatten, Dropout, Convolution2D, Activation, MaxPooling2D, Conv2D, BatchNormalization, \
    ZeroPadding2D
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2

# initialize model
model = models.Sequential()


# training model function
def run_model():
    train = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, shear_range=0.2, zoom_range=0.2,
                               width_shift_range=0.2, height_shift_range=0.2, fill_mode='nearest', validation_split=0.2)

    img_size = 128
    batch_size = 20
    t_steps = 3462 / batch_size
    v_steps = 861 / batch_size
    classes = 5
    flower_path = "C:\\Users\\Yaniv\\Desktop\\flowerClassification-master\\flowers"
    train_gen = train.flow_from_directory(flower_path, target_size=(img_size, img_size), batch_size=batch_size,
                                          class_mode='categorical', subset='training')
    valid_gen = train.flow_from_directory(flower_path, target_size=(img_size, img_size), batch_size=batch_size,
                                          class_mode='categorical', subset='validation')

    model.add(Conv2D(25, kernel_size=(5, 5), activation='relu', input_shape=(128, 128, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # layer number 2
    model.add(Conv2D(50, kernel_size=(5, 5), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # layer number 3
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # layer number 4
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    # layer number 6
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(layers.Dense(classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_hist = model.fit_generator(train_gen, steps_per_epoch=t_steps, epochs=30, validation_data=valid_gen,
                                     validation_steps=v_steps)
    plt_modle(model_hist)
    model.save('flowers_model.h5')
    plt_modle(model_hist)


def plt_modle(model_hist):
    acc = model_hist.history['acc']
    val_acc = model_hist.history['val_acc']
    loss = model_hist.history['loss']
    val_loss = model_hist.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(15, 6));
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, color='#0984e3', marker='o', linestyle='none', label='Training Accuracy')
    plt.plot(epochs, val_acc, color='#0984e3', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, color='#eb4d4b', marker='o', linestyle='none', label='Training Loss')
    plt.plot(epochs, val_loss, color='#eb4d4b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.show()


# Split images into Training and Validation Sets (20%)

def prepare(pathOfFile):
    try:
        img_size = 128
        img = image.load_img(pathOfFile, target_size=(img_size,img_size))
        img = image.img_to_array(img)
        img = np.expand_dims(img,axis = 0)
        #img_array = cv2.imread(pathOfFile)
        #new_img_array = cv2.resize(img_array, (img_size, img_size))
        #return new_img_array.reshape(-1, img_size, img_size, 3)
        return img
    except:
        print("error")


def loading_model(path):
    global model
    model = load_model(path)
    return model


def predict(img):
    global model, categories
    preds = model.predict(img)
    print(preds.argmax(axis=1)[0])
    # print(categories[preds.argmax(axis=1)[0]])
    return categories[preds.argmax(axis=1)[0]]


class Gui:
    def __init__(self, master):
        self.master = master
        master.title("Flower classifier")

        self.label1 = Label(master, text="Choose the trained model")
        self.label1.pack()

        self.button1 = Button(text="Browse Model File", command=browse_button_model)
        self.button1.pack()

        self.label2 = Label(master, text="Choose a folder you would like to classify")
        self.label2.pack()

        self.button2 = Button(text="Browse Image Folder", command=browse_button_img)
        self.button2.pack()

        self.label2 = Label(master, text="Or enter the path")
        self.label2.pack()

        global e1
        e1 = Entry(master)
        e1.pack()

        self.start_classification = Button(master, text="Run Classification", command=start_classification)
        self.start_classification.pack()

        self.label3 = Label(master, text="Train a model")
        self.label3.pack()

        self.run_model = Button(master, text="Train model", command=run_model)
        self.run_model.pack()

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()


def browse_button_img():
    # Allow user to select a directory and store it in global var
    # called folder_path
    global folder_path, folder_c
    filename = filedialog.askdirectory()
    folder_path = filename
    if filename is not "":
        folder_c = True
    else:
        folder_c = False


def browse_button_model():
    # Allow user to select a directory and store it in global var
    # called folder_path
    global model_path, model_c
    filename = filedialog.askopenfilename()
    model_path = filename
    file_ext = filename.split('/')[-1].split('.')[-1]
    if file_ext == "h5":
        model_c = True
        return loading_model(model_path)
        print('loaded model')
    else:
        model_c = False
        print('not loaded')


def start_classification():
    global e1, folder_path
    if model_c and folder_c or model_c and e1.get() != "":
        if e1.get() != "":
            folder_path = e1.get()
        csvfile = open('classification.csv', 'w')
        for subdir, dirs, files in os.walk(folder_path):
            print("Folder: " + subdir)
            for file in files:
                img = prepare(os.path.join(subdir, file))
                line = file + ', ' + predict(img)
                csvfile.write(line + '\n')
        csvfile.close()
    else:
        messagebox.showwarning("Error", "Please choose path of Model and a choose path for the folder/type a path for the folder")


root = Tk()
# Globals
e1 = Entry
categories = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
model_path = StringVar()
folder_path = StringVar()
model_c = False
folder_c = False
#
root.geometry("400x250")
root.resizable(0,0)
my_gui = Gui(root)
root.mainloop()

from tkinter import Tk, Label, Button, filedialog, StringVar

from keras import layers
from keras import models
from keras.layers import Dense, Flatten, Dropout, Convolution2D, Activation, MaxPooling2D, Conv2D, BatchNormalization
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
    train_gen = train.flow_from_directory(flower_path, target_size = (img_size, img_size), batch_size = batch_size, class_mode='categorical', subset='training')
    valid_gen = train.flow_from_directory(flower_path, target_size = (img_size, img_size), batch_size = batch_size, class_mode = 'categorical', subset='validation')

    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(128, 128, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(512, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(layers.Dense(classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_hist = model.fit_generator(train_gen, steps_per_epoch=t_steps, epochs= 40, validation_data=valid_gen, validation_steps=v_steps)
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
def main():
    run_model()

def prepare(pathfile):
    img_size = 128
    img_array = cv2.imread(pathfile)
    new_img_array = cv2.resize(img_array, (img_size,img_size))
    return new_img_array.reshape(-1,img_size,img_size,1)

def load_model(path):
    model.load_model(path)

def predict(path):
    categories = ["daisy","dandelion","rose","sunflower","tulip"]
    preds = model.predict([r'C:\Users\Yaniv\Desktop\flowerClassification-master\flowers\daisy'])
    print(categories[int(preds[0][0])])

class Gui:
    def __init__(self, master):
        self.master = master
        master.title("Flower classifier")

        self.label2 = Label(master, text="Choose the trained model")
        self.label2.pack()

        self.button2 = Button(text="Browse Model File", command=browse_button_model)
        self.button2.pack()

        self.label1 = Label(master, text="Choose a folder you would like to classify")
        self.label1.pack()

        self.button1 = Button(text="Browse Image Folder", command=browse_button_img)
        self.button1.pack()

        self.start_classification = Button(master, text="Run Classification", command=start_classification)
        self.start_classification.pack()

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()

def browse_button_img():
    # Allow user to select a directory and store it in global var
    # called folder_path
    global folder_path
    filename = filedialog.askdirectory()
    folder_path=filename
    print(folder_path)

def browse_button_model():
    # Allow user to select a directory and store it in global var
    # called folder_path
    global model_path
    filename = filedialog.askopenfilename()
    model_path = filename
    print(filename)
    #load_model(filename)

def start_classification():
    split_model_path = model_path.split('/')
    if split_model_path[split_model_path.length - 1] != "":
        print('ahla')


main()
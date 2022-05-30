# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages

from pyimagesearch.TrafficSignNet import TrafficSignNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
from skimage import transform
from skimage import exposure
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_split(basePath, csvPath):
    data = []
    labels = []

    rows = open(csvPath).read().strip().split("\n")[1:]
    random.shuffle(rows)


    for (i, row) in enumerate(rows):
        if i > 0 and i % 1000 == 0:
            print("[INFO] process {} total images".format(i))
        
        (label, imagePath) = row.strip().split(",")[-2:]

        imagePath = os.path.sep.join([basePath, imagePath]).replace("\\","/")
        # Reading image
        image = io.imread(imagePath)

        # Cleaning data
        image = transform.resize(image, (32,32))
        image = exposure.equalize_adapthist(image, clip_limit = 0.1)

        data.append(image)
        labels.append(int(label))

    data = np.array(data)
    labels = np.array(labels)

    return (data, labels)

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input GTRSB")
ap.add_argument("-m","--model", required=True,   help="path to output model")
ap.add_argument("-p","--plot",type=str, default="plot.png",help="path to training history plot")
args = vars(ap.parse_args())

NUM_EPOCH = 10
INIT_LR = 1e-3
BS = 64

labelNames = open("signnames.csv").read().strip().split("\n")[1:]
labelNames = [label.split(",")[1] for label in labelNames]

# derive the path to the training and testing CSV files
trainPath = os.path.sep.join([args["dataset"], 'Train.csv']).replace("\\","/")
testPath = os.path.sep.join([args["dataset"], 'Test.csv']).replace("\\","/")


print("[INFO] loading training and testing data...")

trainX, trainY = load_split(args["dataset"], trainPath)
testX, testY = load_split(args["dataset"], testPath)

trainX = trainX.astype("float32")/255.0
testX = testX.astype("float32")/255.0

numLabels = len(np.unique(trainY))
print(numLabels)

trainY = to_categorical(trainY, numLabels)
testY = to_categorical(testY, numLabels)

print("Length of TrainX: " + str(len(trainX)))
print("Length of TrainY: " + str(len(trainY)))

classTotals = trainY.sum(axis=0)
classWeight = dict()

for i in range(0, len(classTotals)):
    classWeight[i] = classTotals.max() / (classTotals[i])



# construct the image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=10,
	zoom_range=0.15,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.15,
	horizontal_flip=False,
	vertical_flip=False,
	fill_mode="nearest")
# initialize the optimizer, early-stopping and compile the model
print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / (NUM_EPOCH * 0.5))
es = EarlyStopping(monitor='val_loss',patience=2, verbose=1, mode='min')
mc = ModelCheckpoint(args['model'], monitor='val_loss', save_best_only=True)


model = TrafficSignNet.build(width=32, height=32, depth=3,classes=numLabels)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# train the network
print("[INFO] training network...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=trainX.shape[0] // BS,
	epochs=NUM_EPOCH,
	class_weight=classWeight,
	verbose=1,
    callbacks=[es,mc])


print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"])

# save the network to disk

N = np.arange(0, NUM_EPOCH)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

sourcefile = open('summary.txt','w')
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames),file=sourcefile)












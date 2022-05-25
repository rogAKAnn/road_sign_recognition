# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from pyimagesearch.TrafficSignNet import TrafficSignNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from skimage import transform
from skimage import exposure
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_split(basePath, csvPath):
    data = []
    label = []

    rows = open(csvPath).read().strip().split("\n")[1:]
    random.shuffle(rows)

    print(rows[:3])

load_split('','dataset/Train.csv')
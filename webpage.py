import streamlit as st

import numpy as np
import pandas as pd
import os
import torch
import cv2
import random


path = 'saved_weights_emotions.pt'
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        
        self.convolutional_layer = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding = 2),
            nn.ReLU()
        )

        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=3200, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=256, out_features=7)
        )


    def forward(self, x):
        x = self.convolutional_layer(x)
        x = torch.flatten(x, 1)
        x = self.linear_layer(x)
        x = F.softmax(x, dim=1)
        return x

model = CNN()
model.load_state_dict(torch.load(path, map_location = 'cpu'))
for params in model.parameters():
  	params.requires_grad = False

nav = st.sidebar.radio('Navigation', ['Home', 'Recommender', 'Music List', 'About Me'])

if nav =='Home':
	st.title('Emotion Based Music Recommender')
	st.image('front.jpg')
	st.text('Music is the language of all')

if nav =='Recommender':
	st.title('Recommender')
	file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])
	# Load the cascade
	face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
	try:
		from PIL import Image
		image = np.asarray(Image.open(file))
		gray = cv2.cvtColor(np.uint8(image), cv2.COLOR_BGR2GRAY)
		    # Detect the faces
		faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 2)
		    # Draw the rectangle around each face
		st.image(image)    
		for (x, y, w, h) in faces:
		     face = image[y:y + h, x:x + w]

		gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
		st.image(gray_face)
		def rescale_frame(frame):
	   		dim = (48, 48)
	   		return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

		rez_gray_face = rescale_frame(gray_face)

		test_img = torch.tensor(rez_gray_face).reshape((1,1,48,48))/255
		with torch.no_grad():
	         preds = model(test_img)

		st.markdown(preds.tolist())
		final = np.argmax(preds.tolist(), axis = 1)
		final = final[0]
		emotions = {0:'Angry', 1:'Disgusted', 2:'Afraid', 3:'Happy', 4:'Sad', 5:'Surprised', 6:'Neutral'}
		st.write('You are {:>2}% {:}'.format((preds.tolist())[0][final]*100, emotions[final]))
		emo = emotions[final]
		music = pd.read_excel('music.xlsx')
		recom = music[music['Emotion'] == emo]
		recom = recom.reset_index()
		ind = random.randint(0,9)
		st.write('\nNow Playing :', recom.iloc[[ind]]['Title'][ind])
		link = recom.iloc[[ind]]['Link'][ind]
		st.video(link) 
	except:
		st.write('No Face Detected / Waiting for File')	

if nav == 'Music List':
	st.title('Entire Music List')
	music_all= pd.read_excel('music.xlsx')
	st.dataframe(data = music_all)

if nav == 'About Me':
	st.write('This project is created by Mihir Nandawat \n Student of IIT Bombay')


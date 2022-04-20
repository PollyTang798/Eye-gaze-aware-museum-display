
import speech_recognition as sr
import cv2
import pyttsx3
import numpy as np
from playsound import playsound
# import simpleaudio as sa
import os
from ffpyplayer.player import MediaPlayer
from time import sleep
from datetime import datetime
import webbrowser
from PIL import Image
from tkinter import *
from PIL import Image, ImageTk
import argparse
import csv
import gtts
#from infer import gaze_vec, tester

r = sr.Recognizer()
engine = pyttsx3.init('dummy')
# object = ""
endreply = False

def PlayVideo(video_path):
  video=cv2.VideoCapture(video_path)
  player = MediaPlayer(video_path)
  while True:
    grabbed, frame=video.read()
    audio_frame, val = player.get_frame()
    if not grabbed:
      print("End of video")
      break
    if cv2.waitKey(28) & 0xFF == ord("q"):
      break
    cv2.imshow("Video", frame)
    if val != 'eof' and audio_frame is not None:
      #audio
      img, t = audio_frame
  video.release()
  # cv2.destroyAllWindows("Video")

def speak(text):
  # engine.say(text)
  # engine.runAndWait()
  tts = gtts.gTTS(text=text, lang='en')
  filename = "abc.mp3"
  tts.save(filename)
  playsound(filename)
  os.remove(filename)

def recognize_voice():
  text = ''
  with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source)
    voice = r.listen(source)
    try:
      text = r.recognize_google(voice)
    except sr.RequestError:
      speak("Sorry, the I can't access the Google API...")
    except sr.UnknownValueError:
      print("nNt heard")
      return "NA"
  return text.lower()

def reply(text_version):

  print(text_version)

  if "where" in text_version:
    speak("You are in the museum")

  # img = cv2.imread("/Users/TakemiMie/Downloads/TBCS.png")
  # cv2.imshow("test image", img)

  if "introducing" in text_version:
    speak("I am introducing ", object)
  
  if "photo" in text_version or "image" in text_version or "picture" in text_version:
    if object == "laptop":
      speak("showing laptop photo")
      im1 = Image.open("laptop.jpg")
      im1.show()

    elif object == "smartphone":
      speak("showing smartphone photo")
      im2 = Image.open("smartphone.jpg")
      im2.show()

    elif object == "mouse":
      speak("showing mouse photo")
      im3 = Image.open("mouse.jpg")
      im3.show()

  if "development" in text_version:
    if object == "laptop":
      speak("Let me tell you the development of laptop")
    elif object == "smartphone":
      speak("let me tell you the development of smartphone")
    elif object == "mouse":
      speak("let me tell you the development of mouse")

  if object == "smartphone" and "company" in text_version: # specific conversation of one exhibit
    speak("Apple, Samsung, Huawei are some of the main companies producing smartphone nowadays")
  # more specific conversation can be written in similar format

  if "quit" in text_version or "exit" in text_version or "bye" in text_version or "good bye" in text_version :
    speak("It is my pleasure serving you. Have a nice day visiting the museum")
    exit()

  if object == "smartphone" and "video" in text_version:
    speak("Here is a video of smartphone")
    # cv2.destroyWindow('car image')
    PlayVideo("smartphone_video.mp4")
    
  #   # if "play music" in text_version:
  #   #   s_musicfile = "/Users/TakemiMie/Downloads/Eden_EXCEED.wav"
  #   #   playsound(s_musicfile)
    
sleep(1)

def assistant():
  speak("Welcome to the museum")
  if object == "laptop":
    speak("you are interested in laptop")
  elif object == "smartphone":
    speak("You are interested in smartphone")
  elif object == "mouse":
    speak("You are interested in mouse")
  else:
    speak("You are not interested in anything")
  print("checkpoint 1")

  while True:
    print("start speaking")
    text_version = recognize_voice()
    if text_version != "" or text_version != "NA":
      reply(text_version)

def object_selection(x, y, object_data):

  # displaying the coordinates
  for row in object_data:
    # for the actual code
    # if x >= row[1] and x <=row[3] and y >=row[2] and y <=row[4]:
      # return row[0]
    # ---
    if x < 0 and y < 0:
      return "laptop"
    elif x > 0 and y > 0:
      return "smartphone"
    elif x > 0 and y < 0:
      return "mouse"

  return "nth"

# dun use file reading
# call the geddnet function

def threshold(object_data):
  checker = False
  debug_counter = 0
  counter = {}
  last_x = 0
  last_y = 0
  while checker == False:
    file = open('eye_gaze.csv')
    type(file)
    csvreader = csv.reader(file)

    for row in csvreader:
      x = float(row[0])
      y = float(row[1])

    object = object_selection(x, y, object_data) # x and y coordinate stored in the file
    if object not in counter:
      counter[object] = 1
    if x != last_x and y != last_y:
      counter[object] += 1
      last_x = x
      last_y = y

      for object in counter:
        # add a monitor the check if the file contain a new data set
        # if counter[object] > 20: # actual code
        if counter[object] > 0: # debug code
          return object
        if object == "nth":
          for minus in counter:
            if counter[minus] < 0:
              counter[minus] = 0
            else:
              counter[minus] -= 1

  return "nth"

def pre_object():
  file = open('yolo.csv')
  type(file)
  csvreader = csv.reader(file)
  rows = []
  object_list = []
  for row in csvreader:
    rows.append(row)
  for row in rows[0:]:    # Skip the header row and convert first values to integers
    row[1] = int(row[1])
    row[2] = int(row[2])
    row[3] = int(row[3])
    row[4] = int(row[4])
    object_list.append(row[0])

  file.close()
  return rows, object_list

while True:
  # im1 = Image.open("/Users/TakemiMie/Downloads/208C7PAABG.3.jpg")
  # im1.show()
  
  # img = cv2.imread("61DXObmlNpL._AC_SY450_.jpg")
  # cv2.imshow("test_image", img)
  # cv2.waitKey(4)
  # cv2.destroyWindow("test_image")

  exhibit = Image.open("yolo.jpg")
  exhibit.show()
  # img = cv2.imread("/Users/TakemiMie/Downloads/yolo.jpg", 1)
  # cv2.imshow('image', img)
  # x, y = gaze_vec()

  object_data, object_list = pre_object()
  object = threshold(object_data)
  f= open(object + ".txt","w+")
  f.close()
  print("you are looking at ", object)

  # actual code
  while object != "nth":
    assistant()
  # actual code

  # cv2.setMouseCallback('image', click_event)
  # cv2.waitKey(0)

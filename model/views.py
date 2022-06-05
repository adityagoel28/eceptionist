from django.shortcuts import render
import cv2
import pyttsx3
import speech_recognition as sr
import os
import sys
from csv import writer
import numpy as np
import pyautogui as p
from PIL import Image
import pandas as pd
import datetime
import time

# Create your views here.

def home(request):
    return render(request, 'index.html')

def faculty(request):
    chk = 0
    names = ['', 'sankalp', 'sandesh', 'srijan', 'suyash', 'aman', 'tushar', 'meraj', 'amaaan', 'vinayak','swarup','kishan']  # names, leave first empty bcz counter starts from 0
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices')
    engine.setProperty('voices', voices[0].id)

    #text to speech
    def speak(audio):
        engine.say(audio)
        engine.runAndWait()

    # to convert voice into text
    def takecommand():
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("LISTENING.....")
            r.pause_thresold = 2
            audio = r.listen(source, timeout=5,phrase_time_limit=5)

        try:
            print("Recognizing....")
            query =r.recognize_google(audio,language='en-in')
            print(f"user said:{query}")

        except Exception as e:
            speak("say that again please....")
            return "none"
        return query

    #wish
    def wish():
        hour = int(datetime.datetime.now().hour)
        if hour >= 0 and hour <= 12:
            speak("good morning")
        elif hour > 12 and hour <= 18:
            speak("good afternoon")
        else:
            speak("good evening")




    def TaskExecutioner():
        p.press('esc')

        if 1:
            query = takecommand().lower()

            if "available" in query:
                cap = cv2.VideoCapture(0)
                while True:
                    ret,img = cap.read()
                    cv2.imshow('webcam',img)
                    k = cv2.waitKey(50)
                    if k==27:
                        break;
                cap.release()
                cv2.destroyAllWindows()



            elif "no " in query:
                speak("thanks for using me sir, have a good day ")
                sys.exit()

            speak("sir, do you have any other work")


    # face recognition
    def facerecognition():
        recognizer = cv2.face.LBPHFaceRecognizer_create()  # Local Binary Patterns Histograms
        recognizer.read("trainer.yml")  # load trained model
        cascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascadePath)  # initializing haar cascade for object detection approach
        df = pd.read_csv("C:/Users/adity/Downloads/minor/StudentDetails/StudentDetails.csv")
        col_names = ['Id', 'Name', 'Date', 'Time']

        attendance = pd.DataFrame(columns=col_names)

        font = cv2.FONT_HERSHEY_SIMPLEX  # denotes the font type

        id = 200  # number of persons you want to Recognize


        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # cv2.CAP_DSHOW to remove warning
        cam.set(3, 640)  # set video FrameWidht
        cam.set(4, 480)  # set video FrameHeight

        # Define min window size to be recognized as a face
        minW = 0.1 * cam.get(3)
        minH = 0.1 * cam.get(4)

        # flag = True

        if 1:

            ret, img = cam.read()  # read the frames using the above created object

            converted_image = cv2.cvtColor(img,   cv2.COLOR_BGR2GRAY)  # The function converts an input image from one color space to another

            faces = faceCascade.detectMultiScale(
                converted_image,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH)),
            )

            for (x, y, w, h) in faces:

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # used to draw a rectangle on any image

                id, accuracy = recognizer.predict(converted_image[y:y + h, x:x + w])  # to predict on every single image

                # Check if accuracy is less them 100 ==> "0" is perfect match
                if (accuracy < 90):



                    accuracy = "  {0}%".format(round(100 - accuracy))
                    speak("verification successfull")

                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

                    with open('StudentDetails/xyz.csv', 'a') as f_object:

                        row = [id, names[id], date, timeStamp]
                        writer_object = writer(f_object)

                        writer_object.writerow(row)

                        # f_object.close()


                    id = names[id]



                    wish()
                    speak("welcome back")
                    speak(id)
                    speak(" I am a bot and I will be your personal assistant for now how can i help you")
                    TaskExecutioner()

                else:
                    id = "unknown"
                    accuracy = "  {0}%".format(round(100 - accuracy))
                    speak(" verification unsucessful")
                    speak("registering yourself")
                    register()
                    facerecognition()

                cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                cv2.putText(img, str(accuracy), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

            cv2.imshow('camera', img)

            k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video


        cam.release()
        cv2.destroyAllWindows()



    # sample_generator
    def samplegenerator():
        chk =2
        cam = cv2.VideoCapture(0,
                            cv2.CAP_DSHOW)  # create a video capture object which is helpful to capture videos through webcam
        cam.set(3, 640)  # set video FrameWidth
        cam.set(4, 480)  # set video FrameHeight

        detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # Haar Cascade classifier is an effective object detection approach
        face_id=1
        face_id = input("enter face id ")
        # Use integer ID for every new face (0,1,2,3,4,5,6,7,8,9........)

        print("Taking samples, look at camera ....... ")
        speak("Taking samples, look at camera ....... ")
        count = 0  # Initializing sampling face count

        while True:

            ret, img = cam.read()  # read the frames using the above created object
            converted_image = cv2.cvtColor(img,
                                        cv2.COLOR_BGR2GRAY)  # The function converts an input image from one color space to another
            faces = detector.detectMultiScale(converted_image, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # used to draw a rectangle on any image
                count += 1

                cv2.imwrite("samples/face." + str(face_id) + '.' + str(count) + ".jpg", converted_image[y:y + h, x:x + w])
                # To capture & Save images into the datasets folder

                cv2.imshow('image', img)  # Used to display an image in a window

            k = cv2.waitKey(100) & 0xff  # Waits for a pressed key
            if k == 27:  # Press 'ESC' to stop
                break
            elif count >= 10:  # Take 50 sample (More sample --> More accuracy)
                break

        print("Samples taken now closing the program....")
        cam.release()
        cv2.destroyAllWindows()

    def model_trainer():
        path = 'samples'  # Path for samples already taken

        recognizer = cv2.face.LBPHFaceRecognizer_create()  # Local Binary Patterns Histograms
        detector = cv2.CascadeClassifier(
            "haarcascade_frontalface_default.xml")  # Haar Cascade classifier is an effective object detection approach

        def Images_And_Labels(path):  # function to fetch the images and labels

            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            faceSamples = []
            ids = []

            for imagePath in imagePaths:  # to iterate particular image path

                gray_img = Image.open(imagePath).convert('L')  # convert it to grayscale
                img_arr = np.array(gray_img, 'uint8')  # creating an array

                id = int(os.path.split(imagePath)[-1].split(".")[1])
                faces = detector.detectMultiScale(img_arr)

                for (x, y, w, h) in faces:
                    faceSamples.append(img_arr[y:y + h, x:x + w])
                    ids.append(id)

            return faceSamples, ids

        print("Training faces. It will take a few seconds. Wait ...")

        faces, ids = Images_And_Labels(path)
        recognizer.train(faces, np.array(ids))

        recognizer.write('trainer/trainer.yml')  # Save the trained model as trainer.yml

        print("Model trained, Now we can recognize your face.")


    def register():
        speak("Enter your ID: ")
        id = input('Enter your id: ')
        speak("Enter your name: ")
        name = input('Enter your name: ')
        names.append(name)
        speak("Please face the camera capturing images")
        samplegenerator()
        speak("Registering yourself please wait for a moment")
        model_trainer()
        speak("Registerinng successfull thankyou for your patiece")


    facerecognition()
    if(chk == 0):
        return render(request, 'faculty.html')
    else:
        return render(request, 'page2.html')
    

def student(request):
    
    return render(request, 'student.html')

def page2(request):
    return render(request, 'page2.html')

def register(request):
    return render(request, 'test.html')

def book(request):
    return render(request, 'test2.html')
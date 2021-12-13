# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
import face_recognition
import econf
from boltiot import Bolt
import requests

model = "shape_predictor_68_face_landmarks.dat"


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model)

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# Read the users data and create face encodings

image = face_recognition.load_image_file("manthan.jpg")
mface_encoding = face_recognition.face_encodings(image)[0]


known_names = ['OWNER']
known_encods = [mface_encoding]


# BOLT IOT module key and mailgun id to send email alert
mybolt=Bolt(econf.API_KEY,econf.DEVICE_ID)

# start the video stream thread
print("[INFO] starting video stream thread...")
fileStream = True
vs = VideoStream(src='http://192.168.43.1:8080/video').start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(1.0)
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

# loop over frames from the video stream
while True:
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small_frame = cv2.resize(rgb, (0, 0), fx=0.25, fy=0.25)
  
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)
    name = "Unknown"
    face_names = []
    for face_encoding in face_encodings:
        #for i in range(len(known_encods)):
          # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(known_encods, face_encoding)

        if match[0]:
             name = known_names[0]

        face_names.append(name)

   
    # detect faces in the grayscale frame
        rects = detector(gray, 0)

    # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                COUNTER += 1

            # otherwise, the eye aspect ratio is not below the blink
            # threshold
            else:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                # reset the eye frame counter
                COUNTER = 0
            # draw the total number of blinks on the frame along with
            # the computed eye aspect ratio for the frame
            cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    #Authorized= False
   # for n in face_names:

     #     if n != 'Unknown' and TOTAL>2 :
        #           Authorized=True
            
    
    
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            
            if face_names[0]!='Unknown':
               while TOTAL>=3:
              
                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    font = cv2.FONT_HERSHEY_DUPLEX
                     # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom - 35),
                          (right, bottom), (0, 0, 255), cv2.FILLED)
                    
                    cv2.putText(frame, name, (left + 6, bottom - 6),
            
                        font, 1.0, (255, 255, 255), 1)
                    cv2.putText(frame, 'Authorized!!!', (frame.shape[1]//2, frame.shape[0]//2), font, 1.0, (0, 255, 0), 1)
                    break
               
            else:
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, 'Intruder!!!', (frame.shape[1]//2, frame.shape[0]//2), font, 1.0, (0, 0, 255), 1)
                
                cv2.imwrite('opencv_Intruder.jpg',
frame[right:right+left,top:top+bottom])
                
                print("\n[WARNING] Sending alert to owner INTRUDER DETECTED---------------")
                response=mybolt.digitalWrite('0','HIGH')
               
                print("\n [WARNING] Sending e-mail to owner")
                requests.post(
                       "https://api.mailgun.net/v3/sandboxf53b98baec3242cea3eca12639633974.mailgun.org/messages",
                                                                                               auth=("api","d0ada34b8896c5887d08a2c416ef5b05-2ae2c6f3-63fac1fc"),
                       files=[("inline", open("opencv_Intruder.jpg",'rb'))],
                       data={"from": "Home Alert <mailgun@sandboxf53b98baec3242cea3eca12639633974.mailgun.org>",
                           "to":"mannu.manthan@gmail.com",
                           "subject":"Alert Intruder",
                           "text":"someone tried to access!!",
                           "html":'<html>Inline image here:<img src="cid:opencv_Intruder.jpg"></html>'})
            response=mybolt.digitalWrite('0','LOW')
    
    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
print("\n [INFO] Exiting Program and cleaning stuff")
cv2.destroyAllWindows()
vs.stop()

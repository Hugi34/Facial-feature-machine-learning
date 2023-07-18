#Made by Hugo D Leyva 
#detects faces and eyes from live feed or from photos


import cv2

face_detection_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #taking in data on faces
eye_detection_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml') #taking in data on faces


image_path = 'family.jpg' #image to be detected
img = cv2.imread(image_path) #inserting image
video_in = cv2.VideoCapture(1)#takes in camera feed
if not video_in.isOpened():#default function that shoes camera is not loading in
 print("Cannot open camera")
 exit()

def convert_face (video_in): #this function convers frame into greyscale
    gray_scale = cv2.cvtColor(video_in, cv2.COLOR_BGR2GRAY) #converting to grey scale to allow better detection
    face_detection = face_detection_data.detectMultiScale(gray_scale, scaleFactor=1.1, minNeighbors=5, minSize=(40,40) ) #applying face detection
    eye_detection = eye_detection_data.detectMultiScale(gray_scale, scaleFactor=1.1, minNeighbors=5, minSize=(40,40) ) #applying eye detection

    for(x, y, w, h) in face_detection: #this is the converting data for the face
        cv2.rectangle(video_in, (x ,y), (x+w, y+h), (255, 0, 0), 3)#overlaying converted data to make box around face

    for(x, y, w, h) in eye_detection: #this is the converting data for eyes
        cv2.circle(video_in, (x + 40  ,y + 40), (10), (0, 0, 255), 3)#overlaying converted data to make box around eyes
    return face_detection, eye_detection

while True: #while loop to keep photo displayed
    ret, frame = video_in.read()
    if ret is False:
       print("ret is false")
    face_box, eye_box  = convert_face(frame)#sends fram to convertface, which converts to grey scale
    cv2.imshow("face", frame) #displaying photo

    #this refreshs at 1ms
    if cv2.waitKey(1) == ord('q'):
        break
video_in.release()#ends camera feed
cv2.destroyAllWindows()#closes whindow when done 





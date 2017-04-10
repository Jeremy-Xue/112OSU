import pyaudio
import aubio
from aubio import source, tempo, onset
from numpy import median, diff
import subprocess
import wave
import sys
import os
import random
import math
import numpy as np
import argparse
import cv2
import imutils
import threading

# Original code taken from pyaudio sample code, I reformatted it into an audio
# object so that you can create and modify instances of files
# as well as to keep track of the current frame and difficulty
######################Audiofile class######################
class AudioFile:
    def __init__(self, file):
        #init
        self.chunk = 1024
        self.file = file
        self.beats = []
        self.wf = wave.open(file, 'rb')
        self.p = pyaudio.PyAudio()
        self.data = self.wf.readframes(self.chunk)
        self.onsetFrames = []
        self.currentFrame = 0
        self.minChunkSize = 32768
        self.stream = self.p.open(
            format = self.p.get_format_from_width(self.wf.getsampwidth()),
            channels = self.wf.getnchannels(),
            rate = self.wf.getframerate(),
            output = True
        )
    #I initially created a playsegment function, but realized that
    #the beat onset detection worked smoother by just playing the song
    #continuously instead of a set number of chunks
    def play(self):
        #plays entire file
        data = self.wf.readframes(self.chunk)
        while (len(data) > 0):
            self.stream.write(data)
            data = self.wf.readframes(self.chunk)
            self.currentFrame += self.chunk


##########Create audiofile objects (default)###########
closer = AudioFile("/Users/jeremy/Desktop/sounds/closer.wav")
shelter = AudioFile("/Users/jeremy/Desktop/sounds/shelter.wav")
friends = AudioFile("/Users/jeremy/Desktop/sounds/friends.wav")

############Analyze Audiofile################
#Code taken and adapted from aubio python examples.
#Code heavily adapted, collects onset FRAMES instead of SECONDS
#In order to work with my project and how I tracked 
#Beat onsets with my note generation
def analyzeAudio(file, pickedCloser, pickedShelter, pickedFriends):
    win_s = 512                 # fft size
    hop_s = win_s // 2          # hop size

    filename = file

    samplerate = 44100
    if len( sys.argv ) > 2: samplerate = int(sys.argv[2])

    s = source(filename, samplerate, hop_s)
    samplerate = s.samplerate

    o = onset("default", win_s, hop_s, samplerate)

    # list of onsets, in samples
    onsetFrames = []
    # total number of frames read
    total_frames = 0
    while True:
        samples, read = s()
        if o(samples):
            onsetFrames.append(total_frames)
        total_frames += read
        if read < hop_s: break
    if(pickedCloser):
        closer.onsetFrames = onsetFrames
    elif(pickedShelter):
        shelter.onsetFrames = onsetFrames
    elif(pickedFriends):
        friends.onsetFrames = onsetFrames

######################OPENCV OBJECT TRACKING##################
#Object tracking code adapted from 
#http://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
#I modified it to include a game interface and understood how the code worked
#from looking at the OpenCV docs


# THIS IS A REALLY LONG FUNCTION ---BECAUSE--- EVERYTHING HAD TO RUN
# WITHIN THE WHILE LOOP FOR OPENCV TO WORK
def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
        help="path to the (optional) video file")
    ap.add_argument("-b", "--buffer", type=int, default=32,
        help="max buffer size")
    args = vars(ap.parse_args())

    # define the lower and upper boundaries of the "green"
    # ball in the HSV color space, then initialize the
    # list of tracked points
    greenLower = (29, 86, 6)
    greenUpper = (64, 255, 255)
    camera = cv2.VideoCapture(0)
    circleX = random.randint(30, 570)
    circleY = random.randint(30, 270)
    circleR = 30
    nextCircleX = random.randint(30, 570)
    nextCircleY = random.randint(30, 270)
    score = 0
    combo = 1.0
    streak = 0
    rad = 0
    percent = 100.0
    totalNotes = 0
    notesHit = 0
    hit = False

    #SPLASHSCREENVARIABLES
    showSplash = True
    showSongChooser = False
    showDifficultyChooser = False
    showHelp = False
    gameStarted = False
    startGame = False
    startGameCount = 0

    #Default song to Closer
    pickedCloser = True
    pickedShelter = False
    pickedFriends = False
    analyzeAudio(closer.file, True, False, False)
    previousFrame = 0

    # keep looping
    while True:
        # grab the current frame
        (grabbed, frame) = camera.read()
        frame = cv2.flip(frame,1)

        # resize the frame, blur it, and convert it to the HSV
        # color space
        frame = imutils.resize(frame, width=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, greenLower, greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = (500, 500)

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                    (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

        #Splash Screen
        if(showSplash == True):
            cv2.putText(frame, "112 Osu!", (10, 120), cv2.FONT_HERSHEY_DUPLEX,
             4, (255, 255, 255), 3)
            rgbVal = (random.randint(0, 255), random.randint(0, 255),
             random.randint(0, 255))
            cv2.putText(frame, "112 Osu!", (10, 120), cv2.FONT_HERSHEY_DUPLEX,
             4, (rgbVal[0], rgbVal[1], rgbVal[2]), 1)
            cv2.rectangle(frame, (20, 240), (180, 320), (255, 255, 255), -1)
            cv2.rectangle(frame, (20, 240), (180, 320), (0, 0, 0), 2)
            cv2.putText(frame, "Songs", (50, 290), cv2.FONT_HERSHEY_DUPLEX, 1,
             (0, 0, 0), 1)
            if(20 <= center[0] <= 180 and 240 <= center[1] <= 320):
                showSongChooser = True
                showSplash = False
            cv2.rectangle(frame, (220,  240), (380, 320), (255, 255, 255), -1)
            cv2.rectangle(frame, (220,  240), (380, 320), (0, 0, 0), 2)
            cv2.putText(frame, "Difficulty", (225, 290),
             cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
            if(220 <= center[0] <= 380 and 240 <= center[1] <= 320):
                showDifficultyChooser = True
                showSplash = False
            cv2.rectangle(frame, (420, 240), (580, 320), (255, 255, 255), -1)
            cv2.rectangle(frame, (420,  240), (580, 320), (0, 0, 0), 2)
            cv2.putText(frame, "Help", (460, 290), cv2.FONT_HERSHEY_DUPLEX, 1,
             (0, 0, 0), 1)
            if(420 <= center[0] <= 580 and 240 <= center[1] <= 320):
                showHelp = True
                showSplash = False
            cv2.rectangle(frame, (180, 160), (420, 220), (255, 255, 255), -1)
            cv2.rectangle(frame, (180, 160), (420, 220), (0, 0, 0), 2)
            cv2.putText(frame, "Play", (240, 210), cv2.FONT_HERSHEY_DUPLEX, 2,
             (0, 0, 0), 2)
            if(180 <= center[0] <= 420 and 160 <= center[1] <= 220):
                startGame = True
                gameStarted = True
                showSplash = False

        if(showSongChooser == True):
            cv2.rectangle(frame, (20, 20), (180, 140), (255, 255, 255), -1)
            cv2.rectangle(frame, (20, 20), (180, 140), (0, 0, 0), 2)
            cv2.putText(frame, "Closer", (40, 120), cv2.FONT_HERSHEY_DUPLEX, 1,
             (0, 0, 0), 1)
            if(20 <= center[0] <= 180 and 20 <= center[1] <= 140):
                pickedCloser = True
                pickedShelter = False
                pickedFriends = False
                analyze = threading.Thread(target = analyzeAudio,
                 args=(closer.file, True, False, False))
                analyze.start()
                showSongChooser = False
                showSplash = True
            cv2.rectangle(frame, (220,  20), (380, 140), (255, 255, 255), -1)
            cv2.rectangle(frame, (220,  20), (380, 140), (0, 0, 0), 2)
            cv2.putText(frame, "Shelter", (240, 120), cv2.FONT_HERSHEY_DUPLEX,
             1, (0, 0, 0), 1)
            if(220 <= center[0] <= 380 and 20 <= center[1] <= 140):
                pickedCloser = False
                pickedShelter = True
                pickedFriends = False
                analyze = threading.Thread(target = analyzeAudio,
                 args=(shelter.file, False, True, False))
                analyze.start()
                showSongChooser = False
                showSplash = True
            cv2.rectangle(frame, (420, 20), (580, 140), (255, 255, 255), -1)
            cv2.rectangle(frame, (420,  20), (580, 140), (0, 0, 0), 2)
            cv2.putText(frame, "Friends", (440, 120), cv2.FONT_HERSHEY_DUPLEX,
             1, (0, 0, 0), 1)
            if(420 <= center[0] <= 580 and 20 <= center[1] <= 140):
                pickedCloser = False
                pickedShelter = False
                pickedFriends = True
                analyze = threading.Thread(target = analyzeAudio,
                 args=(friends.file, False, False, True))
                analyze.start()
                showSongChooser = False
                showSplash = True

        if(showDifficultyChooser == True):
            cv2.rectangle(frame, (20, 20), (180, 140), (255, 255, 255), -1)
            cv2.putText(frame, "Easy", (40, 120), cv2.FONT_HERSHEY_DUPLEX,
             1, (0, 0, 0), 1)
            if(20 <= center[0] <= 180 and 20 <= center[1] <= 140):
                if(pickedCloser):
                    closer.minChunkSize = 48000
                elif(pickedShelter):
                    shelter.minChunkSize = 48000
                elif(pickedFriends):
                    friends.minChunkSize = 48000
                showSplash = True
                showDifficultyChooser = False
            cv2.rectangle(frame, (220,  20), (380, 140), (255, 255, 255), -1)
            cv2.putText(frame, "Medium", (240, 120), cv2.FONT_HERSHEY_DUPLEX,
             1, (0, 0, 0), 1)
            if(220 <= center[0] <= 380 and 20 <= center[1] <= 140):
                if(pickedCloser):
                    closer.minChunkSize = 32000
                elif(pickedShelter):
                    shelter.minChunkSize = 32000
                elif(pickedFriends):
                    friends.minChunkSize = 32000
                showSplash = True
                showDifficultyChooser = False
            cv2.rectangle(frame, (420, 20), (580, 140), (255, 255, 255), -1)
            cv2.putText(frame, "Hard", (440, 120), cv2.FONT_HERSHEY_DUPLEX, 1,
             (0, 0, 0), 1)
            if(420 <= center[0] <= 580 and 20 <= center[1] <= 140):
                if(pickedCloser):
                    closer.minChunkSize = 24000
                elif(pickedShelter):
                    shelter.minChunkSize = 24000
                elif(pickedFriends):
                    friends.minChunkSize = 24000
                showSplash = True
                showDifficultyChooser = False

        if(showHelp == True):
            cv2.rectangle(frame, (20, 140), (580, 320), (255, 255, 255), -1)
            msg1 = "Hit the colored note on beat with the track"
            msg2 = "The hollow note will tell you where the next note will be"
            msg3 = "Don't miss notes to retain your combo!"
            cv2.putText(frame, msg1, (40, 200), cv2.FONT_HERSHEY_PLAIN, 1,
             (0, 0, 0), 1)
            cv2.putText(frame, msg2, (40, 240), cv2.FONT_HERSHEY_PLAIN, 1,
             (0, 0, 0), 1)
            cv2.putText(frame, msg3, (40, 280), cv2.FONT_HERSHEY_PLAIN, 1,
             (0, 0, 0), 1)
            cv2.rectangle(frame, (20, 20), (220, 120), (255, 255, 255), -1)
            cv2.putText(frame, "Back", (40, 100), cv2.FONT_HERSHEY_DUPLEX, 1,
             (0, 0, 0), 1)
            if(20 <= center[0] <= 220 and 20 <= center[1] <= 120):
                showSplash = True
                showHelp = False

        if(startGame == True):
            startGameCount += 1
            if(startGameCount < 30):
                cv2.putText(frame, "3", (280, 120), cv2.FONT_HERSHEY_DUPLEX, 4,
                 (255, 255, 255), 3)
            elif(startGameCount < 60):
                cv2.putText(frame, "2", (280, 120), cv2.FONT_HERSHEY_DUPLEX, 4,
                 (255, 255, 255), 3)
            elif(startGameCount < 90):
                cv2.putText(frame, "1!", (280, 120), cv2.FONT_HERSHEY_DUPLEX, 4,
                 (255, 255, 255), 3)
            else:
                if(pickedCloser):
                    t = threading.Thread(target=closer.play)
                elif(pickedShelter):
                    t = threading.Thread(target=shelter.play)
                elif(pickedFriends):
                    t = threading.Thread(target=friends.play)
                t.daemon = True
                t.start()
                gameStarted = True
                startGame = False

        if(gameStarted == True):
            #If the player picked Closer
            if(pickedCloser):
                if(closer.currentFrame > closer.onsetFrames[0]):
                    if(hit == False):
                        combo = 1.0
                        streak = 0
                    totalNotes += 1.0
                    hit = False
                    rad = 0
                    prepRad = 0
                    percent = (notesHit / totalNotes) * 100
                    circleX = nextCircleX
                    circleY = nextCircleY
                    nextCircleX = random.randint(30, 570)
                    nextCircleY = random.randint(30, 270)
                    if(len(closer.onsetFrames) > 2):
                        closer.onsetFrames.pop(0)
                    else:
                        gameStarted = False
                        showScore = True
                    previousFrame = closer.currentFrame
                    nextFrame = closer.onsetFrames[0]
                    prepRad = int((nextFrame-previousFrame) / closer.chunk)
                while(closer.onsetFrames[0] <= previousFrame + 
                    closer.minChunkSize and len(closer.onsetFrames) > 1):
                    closer.onsetFrames.pop(0)
                if(abs(center[0] - circleX) <= circleR and 
                    abs(center[1] - circleY) <= circleR):
                    hit = True
                if(hit == True):
                    if(rad == 0):
                        score += 100 * combo
                        streak += 1
                        notesHit += 1.0
                        if(streak % 10 == 0): combo += 0.1
                    if(0 <= rad <= 20):
                        rad += 2
                        cv2.circle(frame, (circleX, circleY), circleR + rad,
                         (255, 255, 255), 2)
                        cv2.circle(frame, (circleX, circleY), circleR,
                         (255, 184, 99), -1)
                else:
                    cv2.circle(frame, (circleX, circleY), circleR,
                     (255, 184, 99), -1)
                cv2.circle(frame, (nextCircleX, nextCircleY), circleR,
                 (0, 0, 0), 2)
                cv2.circle(frame, (nextCircleX, nextCircleY), circleR - 3,
                 (255, 255, 255), 2)
                cv2.circle(frame, (nextCircleX, nextCircleY), circleR + 3,
                 (255, 255, 255), 2)
                cv2.putText(frame, "score: " + str(int(score)), (10, 30),
                 cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "combo: " + str(combo), (400, 330),
                 cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                if(percent >= 90):
                    pColor = (0, 255, 0)
                elif(percent >= 70):
                    pColor = (0, 255, 255)
                elif(percent >= 60):
                    pColor = (0, 0, 255)
                else:
                    showScore = True
                    gameStarted = False
                cv2.putText(frame, "Percent: %.2f" % percent, (10, 330),
                 cv2.FONT_HERSHEY_DUPLEX, 1, (pColor[0], pColor[1], pColor[2]),
                  2)
                cv2.putText(frame, "x" + str(int(streak)), (520, 30),
                 cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

            #Otherwise if the player picked shelter
            elif(pickedShelter):
                if(shelter.currentFrame > shelter.onsetFrames[0]):
                    if(hit == False):
                        combo = 1.0
                        streak = 0
                    totalNotes += 1.0
                    hit = False
                    rad = 0
                    prepRad = 0
                    percent = (notesHit / totalNotes) * 100
                    circleX = nextCircleX
                    circleY = nextCircleY
                    nextCircleX = random.randint(30, 570)
                    nextCircleY = random.randint(30, 270)
                    if(len(shelter.onsetFrames) > 2):
                        shelter.onsetFrames.pop(0)
                    else:
                        gameStarted = False
                        showScore = True
                    previousFrame = shelter.currentFrame
                    nextFrame = shelter.onsetFrames[0]
                    prepRad = int((nextFrame-previousFrame) / shelter.chunk)
                while(shelter.onsetFrames[0] <= previousFrame + 
                    shelter.minChunkSize and len(shelter.onsetFrames) > 1):
                    shelter.onsetFrames.pop(0)
                if(abs(center[0] - circleX) <= circleR and 
                    abs(center[1] - circleY) <= circleR):
                    hit = True
                if(hit == True):
                    if(rad == 0):
                        score += 100 * combo
                        streak += 1
                        notesHit += 1.0
                        if(streak % 10 == 0): combo += 0.1
                    if(0 <= rad <= 20):
                        rad += 2
                        cv2.circle(frame, (circleX, circleY), circleR + rad,
                         (255, 255, 255), 2)
                        cv2.circle(frame, (circleX, circleY), circleR,
                         (255, 184, 99), -1)
                else:
                    cv2.circle(frame, (circleX, circleY), circleR,
                     (255, 184, 99), -1)
                cv2.circle(frame, (nextCircleX, nextCircleY), circleR,
                 (0, 0, 0), 2)
                cv2.circle(frame, (nextCircleX, nextCircleY), circleR - 3,
                 (255, 255, 255), 2)
                cv2.circle(frame, (nextCircleX, nextCircleY), circleR + 3,
                 (255, 255, 255), 2)
                cv2.putText(frame, "score: " + str(int(score)), (10, 40),
                 cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "combo: " + str(combo), (400, 330),
                 cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                if(percent >= 90):
                    pColor = (0, 255, 0)
                elif(percent >= 70):
                    pColor = (0, 255, 255)
                elif(percent >= 60):
                    pColor = (0, 0, 255)
                else:
                    showScore = True
                    gameStarted = False
                cv2.putText(frame, "Percent: %.2f" % percent, (10, 330),
                 cv2.FONT_HERSHEY_DUPLEX, 1, (pColor[0], pColor[1], pColor[2]),
                  2)
                cv2.putText(frame, "x" + str(int(streak)), (520, 30),
                 cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

            #Otherwise if the player picked Friends
            elif(pickedFriends):
                if(friends.currentFrame > friends.onsetFrames[0]):
                    if(hit == False):
                        combo = 1.0
                        streak = 0
                    totalNotes += 1.0
                    hit = False
                    rad = 0
                    prepRad = 0
                    percent = (notesHit / totalNotes) * 100
                    circleX = nextCircleX
                    circleY = nextCircleY
                    nextCircleX = random.randint(30, 570)
                    nextCircleY = random.randint(30, 270)
                    if(len(friends.onsetFrames) > 1):
                        friends.onsetFrames.pop(0)
                    else:
                        gameStarted = False
                        showScore = True
                    previousFrame = friends.currentFrame
                    nextFrame = friends.onsetFrames[0]
                    prepRad = int((nextFrame-previousFrame) / friends.chunk)
                while(friends.onsetFrames[0] <= previousFrame + 
                    friends.minChunkSize and len(friends.onsetFrames) > 1):
                    friends.onsetFrames.pop(0)
                if(abs(center[0] - circleX) <= circleR and 
                    abs(center[1] - circleY) <= circleR):
                    hit = True
                if(hit == True):
                    if(rad == 0):
                        score += 100 * combo
                        streak += 1
                        notesHit += 1.0
                        if(streak % 10 == 0): combo += 0.1
                    if(0 <= rad <= 20):
                        rad += 2
                        cv2.circle(frame, (circleX, circleY), circleR + rad, 
                            (255, 255, 255), 2)
                        cv2.circle(frame, (circleX, circleY), circleR, 
                            (255, 184, 99), -1)
                else:
                    cv2.circle(frame, (circleX, circleY), circleR, 
                        (255, 184, 99), -1)
                cv2.circle(frame, (nextCircleX, nextCircleY), circleR, 
                    (0, 0, 0), 2)
                cv2.circle(frame, (nextCircleX, nextCircleY), circleR - 3,
                 (255, 255, 255), 2)
                cv2.circle(frame, (nextCircleX, nextCircleY), circleR + 3,
                 (255, 255, 255), 2)
                cv2.putText(frame, "score: " + str(int(score)), (10, 40),
                 cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "combo: " + str(combo), (400, 330),
                 cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                if(percent >= 90):
                    pColor = (0, 255, 0)
                elif(percent >= 70):
                    pColor = (0, 255, 255)
                elif(percent >= 60):
                    pColor = (0, 0, 255)
                else:
                    showScore = True
                    gameStarted = False
                cv2.putText(frame, "Percent: %.2f" % percent, (10, 330),
                 cv2.FONT_HERSHEY_DUPLEX, 1, (pColor[0], pColor[1], pColor[2]), 2)
                cv2.putText(frame, "x" + str(int(streak)), (520, 30),
                 cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

    
        # show the frame to our screen
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break
     
    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()

#########################################################################
run()



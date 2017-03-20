# Import the required modules
import dlib
import cv2

inputVideo="/home/sagar/IMAGR/Quiz/QuizVideo.mp4"

# Create the VideoCapture object
cam = cv2.VideoCapture(inputVideo)
points=[]
points.append((645,777,1353,960))
retval, img = cam.read()
# Initial co-ordinates of the object to be tracked
# Create the tracker object
tracker = dlib.correlation_tracker()
# Provide the tracker the initial position of the object
s=points[0]
tracker.start_track(img, dlib.rectangle(*points[0]))
store_cordinate = []
while True:
     # Read frame from device or file
     retval, img = cam.read()
     if not retval:
       print "Cannot capture frame device | CODE TERMINATING :("
       exit()
# Update the tracker
     tracker.update(img)
# Get the position of the object, draw a
# bounding box around it and display it.
     rect = tracker.get_position()
     pt1 = (int(rect.left()), int(rect.top()))
     pt2 = (int(rect.right()), int(rect.bottom()))
     cv2.rectangle(img, pt1, pt2, (0,0,255), 4)
     print "Object tracked at [{}, {}] \r".format(pt1, pt2),
     loc = (int(rect.left()), int(rect.top()-20))
     store_cordinate.append(loc)
     txt = "Object tracked at [{}, {}]".format(pt1, pt2)
     cv2.putText(img, txt, loc , cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),4)
     cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
     cv2.imshow("Image", img)
     # Continue until the user presses ESC key
     if cv2.waitKey(1) == 27:
       break

# Relase the VideoCapture object
cam.release()
print "Print it ",store_cordinate

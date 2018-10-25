# Real-Time-Face-Recognition
Real Time Face Recognition using the face_recognition library and opencv

**Usage**

1. Create a folder called 'faces' and put the pictures of people you want the program to recognize.
2. Rename each picture with the name of the Person in the picture (make sure that person is the only face in the image)
3. Run the program:
    It uses the webcam and detects all the faces in the frame and puts a bounding box around each face and name of the person (from the filename) or 'Unkown' if the person is not found in the database
Note: face_recognition_every_other_frame.py does the detection and recognition process on one frame and then skips one frame to improve speed but does.

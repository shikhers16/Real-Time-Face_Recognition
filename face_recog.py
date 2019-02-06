import cv2
import numpy as np

def who_is_it(image, database, model):
    ##Compute the target "encoding" for the image.
    encoding = img_to_encoding(image, model)
    
    #Find the closest encoding
    # Initializing minimum distance
    min_dist = 100
    #Loop over the database dictionary's names and encodings.
    for (name, enc) in database.items():
        # Compute L2 distance between the target "encoding" and the current encoding from the database.
        dist = np.linalg.norm(encoding - enc)
        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name.
        if dist < min_dist:
            min_dist = dist
            identity = name
    if min_dist > 0.5:
        print("Not in the database.")
        identity = "Unknown"
        color = (255, 0, 0)
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        color = (0, 255, 0)
    return min_dist, identity, color

def img_path_to_encoding(image_path, model):
    img1 = cv2.imread(image_path, 1)
    return img_to_encoding(img1, model)

def img_to_encoding(image, model):
    image = cv2.resize(image, (96, 96)) 
    img = image[...,::-1]
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    embedding = embedding[0]
    return embedding

def display(frame, x, y, w, h, name, color):
        # Draw a box around the face
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        # Draw a label with a name below the face
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    cv2.putText(frame, name, (x + 10, y+h + 20), font, 1, color, 1)

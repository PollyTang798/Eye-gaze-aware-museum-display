import cv2
import numpy as np
import csv

def captureImage():
    cam = cv2.VideoCapture(1, cv2.CAP_DSHOW) #Change to 1 if using external webcam
    image = cam.read()[1]

    return image

def detectObjects():
    #Load YOLOv4 Algorithm
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

    #To load all objects that have to be detected
    classes = []
    with open("coco.names","r") as f:
        read = f.readlines()
    for i in range(len(read)):
        classes.append(read[i].strip("\n"))

    #Defining layer names
    layer_names = []
    layer_names = net.getLayerNames()
    output_layers = []
    for i in net.getUnconnectedOutLayers():
        output_layers.append(layer_names[i-1])


    #Capture an image
    img = captureImage()
    height, width, channels = img.shape


    #Extracting features to detect objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)


    #Displaying informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for output in outs:
        for detection in output:
            #Detecting confidence
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5: #Means if the object is detected
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                #Drawing a rectangle
                x = int(center_x-w/2) # top left value
                y = int(center_y-h/2) # top left value

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
               #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            

    #Remove Double Boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
    labels = []
    obj_boxes = []
    objects = {}

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            obj_boxes.append(boxes[i])
            labels.append(classes[class_ids[i]])  # name of the objects
            objects[classes[class_ids[i]]] = [x,y,w,h]

            #print("Object: ", classes[class_ids[i]]) #, "   ", confidences[i])
            #print("Top left x = ", x)
            #print("Top left y = ", y)
            #print("Width = ", w)
            #print("Height = ", h)

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, classes[class_ids[i]], (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
       

    #print(objects)
    cv2.imshow("Output", img)
    cv2.imwrite("output.jpg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return labels, obj_boxes

#detectObjects()

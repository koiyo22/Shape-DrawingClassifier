from tkinter import *
from tkinter import simpledialog, filedialog
import tkinter.messagebox

import pickle
import os.path
import numpy as np
import PIL
import PIL.Image, PIL.ImageDraw
import cv2 as cv
import random

from sklearn.neighbors import KNeighborsClassifier

class ShapeClassifier:
    #blueprint
    def __init__(self):
        #allow classification of 3 classes
        self.class1, self.class2, self.class3 = None, None, None
        # track how many examples filled for each class
        self.class1Counter, self.class2Counter, self.class3Counter = None, None, None
        
        self.classifier = KNeighborsClassifier(n_neighbors=3) #generally performs better

        self.spaceName=None
        self.root=None
        self.image1=None
        self.heading=None
        self.canvas = None
        self.draw=None
        self.text1=None
        self.text2=None
        self.text3=None

        #brush size
        self.brushWidth=15
        self.classesPrompt()
        self.initGUI()

    def classesPrompt(self):
        message=Tk()
        message.withdraw()

        self.spaceName = simpledialog.askstring("Space Name","Enter a name for your space", parent=message)

        if os.path.exists(self.spaceName):
            with open(f"{self.spaceName}/{self.spaceName}_data.pickle","rb") as file:
                data=pickle.load(file)
            self.class1=data['c1']
            self.class2=data['c2']
            self.class3=data['c3']
            self.class1Counter=data['c1c']
            self.class2Counter=data['c2c']
            self.class3Counter=data['c3c']
            self.classifier=data['classifier']
            self.spaceName=data['spaceName']
        else:
            self.class1=simpledialog.askstring("Shape 1","First shape to classify:",parent=message)
            self.class2=simpledialog.askstring("Shape 2","Second shape to classify:",parent=message)
            self.class3=simpledialog.askstring("Shape 3","Third shape to classify:",parent=message)

            self.class1Counter=1
            self.class2Counter=1
            self.class3Counter=1

            os.mkdir(self.spaceName)
            os.chdir(self.spaceName)
            os.mkdir(self.class1)
            os.mkdir(self.class2)
            os.mkdir(self.class3)
            os.chdir("..")

    def initGUI(self):
        #system GUI creation

        WIDTH=500
        HEIGHT=500
        WHITE=(255,255,255)
        self.root=Tk()
        self.root.title(f'Shape Classifier - {self.spaceName}')
        self.heading=Label(text="Draw more samples for accurate predictions!")

        self.canvas = Canvas(self.root, width=WIDTH-10, height=HEIGHT-10, background="white")
        
        self.canvas.pack(expand = YES, fill =BOTH)

        self.canvas.bind("<B1-Motion>", self.paint) #bind mouse clicking motions

        #counter frame
        counterFrame=Frame(self.root)
        counterFrame.place(relx=0.8,rely=0.05)

        self.text1=Label(counterFrame,text=f"{self.class1}:{self.class1Counter-1}")
        self.text2=Label(counterFrame,text=f"{self.class2}:{self.class1Counter-1}")
        self.text3=Label(counterFrame,text=f"{self.class3}:{self.class1Counter-1}")

        self.text1.pack(anchor=E)
        self.text2.pack(anchor=E)
        self.text3.pack(anchor=E)

        #binding the motion of clicking a button with a function
        self.image1 = PIL.Image.new("RGB", (WIDTH, HEIGHT), WHITE) 
        self.draw = PIL.ImageDraw.Draw(self.image1)

        buttonFrame = tkinter. Frame(self.root) 
        buttonFrame.pack(fill=X, side = BOTTOM) 
        buttonFrame.place(relx=0.5, rely=0.9, anchor=CENTER)

        Button(buttonFrame, text=self.class1, command=lambda:self.save(1)).grid(row=0, column=0, sticky=W+E)
        Button(buttonFrame, text=self.class2, command=lambda:self.save(2)).grid(row=0, column=1, sticky=W+E)
        Button(buttonFrame, text=self.class3, command=lambda:self.save(3)).grid(row=0, column=2, sticky=W+E)
        Button(buttonFrame, text="- Brush Size", command=self.brushminus).grid(row=1, column=0, sticky=W+E)
        Button(buttonFrame, text="Clear Canvas", command=self.clear).grid(row=1, column=1, sticky=W+E)
        Button(buttonFrame, text="+ Brush Size", command=self.brushplus).grid(row=1, column=2, sticky=W+E)
        Button(buttonFrame, text="Train Model", command=self.trainModel).grid(row=2, column=0, sticky=W+E)
        Button(buttonFrame, text="Save Model", command=self.saveModel).grid(row=2, column=1, sticky=W+E)
        Button(buttonFrame, text="Load Model", command=self.loadModel).grid(row=2, column=2, sticky=W+E)
        Button(buttonFrame, text="Predict Current Shape", command=self.predictClass).grid(row=3, column=1, sticky=W+E)

        self.root.protocol("WM_DELETE_WINDOW",self.onClosing)
        self.root.attributes("-topmost", True)
        self.root.mainloop()

    def paint(self,event):
        x1,y1=(event.x -1), (event.y-1)
        x2,y2=(event.x -1), (event.y-1)
        self.canvas.create_oval(x1,y1,x2,y2,fill="red",width=self.brushWidth)

        self.draw.circle([x1, y1, x2 + self.brushWidth, y2 + self.brushWidth], fill="red", radius = self.brushWidth)

   
    def save(self, classNum):
        self.image1.save("temp.png")
        image = PIL.Image.open("temp.png")
        image.thumbnail((50, 50), PIL.Image.LANCZOS)
        
        if classNum == 1:
            save_path = f"{self.spaceName}/{self.class1}/{self.class1Counter}.png"
            self.class1Counter += 1
        elif classNum == 2:
            save_path = f"{self.spaceName}/{self.class2}/{self.class2Counter}.png"
            self.class2Counter += 1
        elif classNum == 3:
            save_path = f"{self.spaceName}/{self.class3}/{self.class3Counter}.png"
            self.class3Counter += 1

        image.save(save_path, "PNG")

        # Generate and save augmented images
        augmented_images = self.augment_image(image)
        for idx, aug_img in enumerate(augmented_images):
            aug_path = save_path.replace(".png", f"_aug{idx+1}.png")
            aug_img.save(aug_path, "PNG")

        self.updateLabels()
        self.clear()

    def augment_image(self, image): #for each image saved -> 1+ 5 augmentations
        augmented_images = []
        img_np = np.array(image)

        #random rotation 
        for angle in [-20, 20]:
            rot_mat = cv.getRotationMatrix2D((img_np.shape[1] // 2, img_np.shape[0] // 2), angle, 1)
            rotated = cv.warpAffine(img_np, rot_mat, (img_np.shape[1], img_np.shape[0]), borderValue=(255,255,255))
            augmented_images.append(PIL.Image.fromarray(rotated))

        #rnadom scaling
        for scale in [0.9, 1.1]:  # zoom out/in
            resized = cv.resize(img_np, (0,0), fx=scale, fy=scale)
            resized = cv.resize(resized, (img_np.shape[1], img_np.shape[0]))  #force resize back to original
            augmented_images.append(PIL.Image.fromarray(resized))

        #translation
        for shift_x in [-5, 5]:  # shift left and right
            M = np.float32([[1, 0, shift_x], [0, 1, 0]])
            translated = cv.warpAffine(img_np, M, (img_np.shape[1], img_np.shape[0]), borderValue=(255,255,255))
            augmented_images.append(PIL.Image.fromarray(translated))

        return augmented_images

    def updateLabels(self):
        self.text1.config(text=f"{self.class1}: {self.class1Counter-1}")
        self.text2.config(text=f"{self.class2}: {self.class2Counter-1}")
        self.text3.config(text=f"{self.class3}: {self.class3Counter-1}")
    
    def trainModel(self):
        imageList = np.array([])
        classList = np.array([])

        for x in range(1, self.class1Counter):
            image = cv.imread(f"{self.spaceName}/{self.class1}/{x}.png")[:,:,0] 
            image = cv.resize(image, (50, 50))
            image = image / 255.0   
            image= image.reshape(2500)
            imageList = np.append(imageList, [image])
            classList = np.append(classList, 1)

        for x in range(1, self.class2Counter):
            image = cv.imread(f"{self.spaceName}/{self.class2}/{x}.png")[:,:,0] 
            image = cv.resize(image, (50, 50))
            image = image / 255.0 
            image =image.reshape(2500)
            imageList = np.append(imageList, [image]) 
            classList = np.append(classList, 2)

        for x in range(1, self.class3Counter):
            image = cv.imread(f"{self.spaceName}/{self.class3}/{x}.png")[:,:,0] 
            image = cv.resize(image, (50, 50))
            image = image / 255.0 
            image =image.reshape(2500)
            imageList = np.append(imageList, [image])
            classList = np.append(classList, 3)
        
        imageList = imageList.reshape(self.class1Counter -1 + self.class2Counter -1 + self.class3Counter - 1, 2500)
        
        self.classifier.fit(imageList, classList)
        tkinter.messagebox.showinfo("Shape Classifier", "Model successfully trained!", parent = self.root)

        
    def saveModel(self):
        #savve trained model as pickle file
        filePath = filedialog.asksaveasfilename (defaultextension="pickle")
        with open(filePath, "wb") as f:
            pickle.dump(self.classifier, f)
        tkinter.messagebox.showinfo("Shape Classifier", f"Model successfully saved!", parent = self.root)

    def loadModel(self):
        filePath = filedialog.askopenfilename()
        with open(filePath, "rb") as f:
            self.classifier = pickle.load(f)
        tkinter.messagebox.showinfo("Shape Classifier", f"Model successfully loaded", parent = self.root)

    def predictClass(self):
        self.image1.save("temp.png")
        image = PIL.Image.open("temp.png") 
        image.thumbnail((50, 50), PIL.Image.LANCZOS)
        image.save("predictClass.png", "PNG")

        image = cv.imread("predictClass.png")[:, :, 0] 
        image = image.reshape(1, 2500)  #flatten to 2D → shape (1, 2500)

        prediction = self.classifier.predict(image)  

        if prediction[0] == 1:
            tkinter.messagebox.showinfo("Shape Classifier", f"This should be a {self.class1}", parent=self.root)
        elif prediction[0] == 2:
            tkinter.messagebox.showinfo("Shape Classifier", f"This should be a {self.class2}", parent=self.root)
        elif prediction[0] == 3:
            tkinter.messagebox.showinfo("Shape Classifier", f"This should be a {self.class3}", parent=self.root)

    def saveEverything(self):
        data = {"c1" : self.class1,
                "c2": self.class2,
                "c3": self.class3,
                "c1c": self.class1Counter, "c2c": self.class2Counter,
                "c3c" :self.class3Counter,
                "classifier": self.classifier, 
                "spaceName" : self.spaceName}
        
        with open(f"{self.spaceName}/{self.spaceName}_data.pickle", "wb") as f:
            pickle.dump(data, f)
        tkinter.messagebox.showinfo("Shape Classifier", "Progress saved!", parent = self.root)
    
    def onClosing(self):
        ans=tkinter.messagebox.askyesnocancel("Quit?","Save ur progress?",parent=self.root)

        if ans is not None:
            if ans:
                self.saveEverything()
            self.root.destroy()
            exit()

    def brushminus(self):
        if self.brushWidth>1:
            self.brushWidth -=2

    def brushplus(self):
        if self.brushWidth<100:
            self.brushWidth +=2

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0,0,800,800],fill="white")


ShapeClassifier()
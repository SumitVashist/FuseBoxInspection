import tkinter as tk
import time
import tkinter.font as font

import numpy as np
import argparse
import time
import cv2
import os

from glob import glob
from datetime import datetime

dispH=720
dispW=720
flip=2
key='0'
camSet='nvarguscamerasrc sensor-id='+key+' ! video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
import pandas

df = pandas.read_excel('employeecode.xlsx')
#print the column names
print (df.columns)
#get the values for a given column
values = df['EMPLOYEE CODE'].values
print(values)

current_balance = 1000

class SampleApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.shared_data = {'Balance':tk.IntVar()}

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage,StartPage1):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent,bg='#3d3d5c')
        self.controller = controller

        self.controller.title('Marj Vision  AI-X ')
        self.controller.state('normal')
        #self.controller.iconphoto(False,tk.PhotoImage(file='C:/Users/urban boutique/Documents/atm tutorial/atm.png'))

        heading_label = tk.Label(self,
                                                     text='Marj Vision  AI-X Inspection ',
                                                     font=('orbitron',40,'bold'),
                                                     foreground='#ffffff',
                                                     background='#3d3d5c')
        heading_label.pack(pady=100)

        space_label = tk.Label(self,height=5,bg='#3d3d5c')
        space_label.pack()

        password_label = tk.Label(self,
                                                      text='Enter your Employee ID',
                                                      font=('orbitron',15),
                                                      bg='#3d3d5c',
                                                      fg='white')
        password_label.pack(pady=9)

        my_password = tk.StringVar()
        password_entry_box = tk.Entry(self,
                                                              textvariable=my_password,
                                                              font=('orbitron',12),
                                                              width=22)
        password_entry_box.focus_set()
        password_entry_box.pack(ipady=7)

        def handle_focus_in(_):
            password_entry_box.configure(fg='black')
            
        password_entry_box.bind('<FocusIn>',handle_focus_in)
		
           


        def check_password():
           print(int(16616) in values)
           if (my_password.get())=='123':
               my_password.set('')
               incorrect_password_label['text']=''
        
               controller.show_frame('StartPage1')

        	   
        
           else:
               incorrect_password_label['text']='Incorrect Id'
                
        enter_button = tk.Button(self,
                                                     text='Enter',
                                                     command=check_password,
                                                     relief='raised',
                                                     borderwidth = 5,
                                                     width=23,
                                                     height=3)
        enter_button.pack(pady=20)

        incorrect_password_label = tk.Label(self,
                                                                        text='',
                                                                        font=('orbitron',13),
                                                                        fg='white',
                                                                        bg='#33334d',
                                                                        anchor='n')
        incorrect_password_label.pack(fill='both',expand=True)

        bottom_frame = tk.Frame(self,relief='raised',borderwidth=3)
        bottom_frame.pack(fill='x',side='bottom')




        def tick():
            current_time = time.strftime('%I:%M %p').lstrip('0').replace(' 0',' ')
            time_label.config(text=current_time)
            time_label.after(200,tick)

            
        time_label = tk.Label(bottom_frame,font=('orbitron',12))
        time_label.pack(side='right')


        tick()


class StartPage1(tk.Frame):

	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent,bg='#3d3d5c')
		self.controller = controller

		self.controller.title('Marj Vision  AI-X ')
		#self.controller.state('iconic')
		#self.controller.pack()
		heading_label = tk.Label(self,text='Marj Vision  AI-X Inspection ',font=('orbitron',45,'bold'),fg='#ffffff',bg='#3d3d5c')
		heading_label.pack(pady=100)

		space_label = tk.Label(self,height=5,bg='#3d3d5c')
		space_label.pack(pady=100)
		i=0
		def check_password():
			camera=cv2.VideoCapture(camSet)
			
			# construct the argument parse and parse the arguments
			ap = argparse.ArgumentParser()
			# ap.add_argument("-i", "--image", required=True, help="path to input image")
			ap.add_argument(
				"-y", "--yolo", required=False, default="yolo-trained-files", help="base path to YOLO directory"
			)
			ap.add_argument(
				"-c",
				"--confidence",
				type=float,
				default=0.0001,
				help="minimum probability to filter weak detections",
			)
			ap.add_argument(
				"-t",
				"--threshold",
				type=float,
				default=0.00001,
				help="threshold when applyong non-maxima suppression",
			)
			args = vars(ap.parse_args())

			# load the Digits class labels our YOLO model was trained on
			labelsPath = os.path.sep.join([args["yolo"], "obj.names"])
			LABELS = open(labelsPath).read().strip().split("\n")

			# initialize a list of colors to represent each possible class label
			np.random.seed(24)
			COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
			black = [0, 0, 0]
			# derive the paths to the YOLO weights and model configuration
			weightsPath = os.path.sep.join([args["yolo"], "best.weights"])
			configPath = os.path.sep.join([args["yolo"], "custom-yolov4-tiny-detector.cfg"])

			# load our YOLO object detector trained on COCO dataset (80 classes)
			print("[INFO] loading Tiny-YOLO V4 from disk...")
			net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
			test=0
			start_time = time.time()
			image_path=''
			
			while test<1:
				succ,image = camera.read()
				#image=image[200:500,230:500]
				cv2.imshow("Image",image)
				if cv2.waitKey(10) & 0xFF == ord('q'):
					break
				if time.time() - start_time >=1:
					dt=str(datetime.now()) #<---- Check if 5 sec passed
					cv2.imwrite('images/'+str(dt)+'.png', image)
					print(dt)
					image_path ='images/'+str(dt)+'.png'
					start_time = time.time()
					test+=1

			image = cv2.imread(image_path)

			(H, W) = image.shape[:2]

			# determine only the *output* layer names that we need from YOLO
			ln = net.getLayerNames()
			ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

			# construct a blob from the input image and then perform a forward
			# pass of the YOLO object detector, giving us our bounding boxes and
			# associated probabilities
			blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
			net.setInput(blob)
			start = time.time()
			layerOutputs = net.forward(ln)
			end = time.time()

			# show timing information on YOLO
			print("[INFO] Inference Time: {:.6f} seconds".format(end - start))

			# initialize our lists of detected bounding boxes, confidences, and
			# class IDs, respectively
			boxes = []
			confidences = []
			classIDs = []

			# loop over each of the layer outputs
			for output in layerOutputs:
				# loop over each of the detections
				for detection in output:
					# extract the class ID and confidence (i.e., probability) of
					# the current object detection
					scores = detection[5:]
					classID = np.argmax(scores)
					confidence = scores[classID]

					# filter out weak predictions by ensuring the detected
					# probability is greater than the minimum probability
					if confidence > args["confidence"]:
						# scale the bounding box coordinates back relative to the
						# size of the image, keeping in mind that YOLO actually
						# returns the center (x, y)-coordinates of the bounding
						# box followed by the boxes' width and height
						box = detection[0:4] * np.array([W, H, W, H])
						(centerX, centerY, width, height) = box.astype("int")

						# use the center (x, y)-coordinates to derive the top and
						# and left corner of the bounding box
						x = int(centerX - (width / 2))
						y = int(centerY - (height / 2))

						# update our list of bounding box coordinates, confidences,
						# and class IDs
						boxes.append([x, y, int(width), int(height)])
						confidences.append(float(confidence))
						classIDs.append(classID)

			# apply non-maxima suppression to suppress weak, overlapping bounding
			# boxes
			idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
			# ensure at least one detection exists
			# print(classIDs)
			# print(idxs.flatten())
			recog_digits = []
			digits_confi = {}
			xcoord_dict = {}
			if len(idxs) > 0:
				# loop over the indexes we are keeping
				for i in idxs.flatten():
					# extract the bounding box coordinates
					(x, y) = (boxes[i][0], boxes[i][1])
					(w, h) = (boxes[i][2], boxes[i][3])

					# draw a bounding box rectangle and label on the image
					color = (0,255,0)
					cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
					output_text = "{}: {:.1%}".format(LABELS[classIDs[i]], confidences[i])
					# print(output_text)

					digits_confi[classIDs[i]] = confidences[i]

					# xcoord_dict[classIDs[i]] = x
					xcoord_dict[x] = classIDs[i]

					text = "{}".format(LABELS[classIDs[i]])
				   # cv2.rectangle(image, (x, y - 30), (x + w, y + h - 35), color, thickness=cv2.FILLED)
					cv2.putText(image, text, (x+1, y+20 ), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

			out_filename = "output/" + image_path.split("/")[-1]
			cv2.imwrite(out_filename, image)
			# xcoord_dict = dict(sorted(xcoord_dict.items(), key=lambda item: item[1]))
			xcoord_dict = dict(sorted(xcoord_dict.items()))

			print("\n[OUTPUT]\nRecognized Digit: Confidence")
			recognized_digits = []
			for key, value in xcoord_dict.items():
				print(value, ":", "{:.1%}".format(digits_confi[value]))
				recognized_digits.append(str(value))

			print("\nFinal Output:")
			output_digits = "".join(recognized_digits)
			print(output_digits)
			cv2.imshow("Image",image)
			cv2.waitKey(10)
			controller.show_frame('StartPage1')
					   
		enter_button = tk.Button(self,text='Inspect',command=check_password,font=('orbitron',45,'bold'),relief='raised',borderwidth = 5,width=30,height=10)
		enter_button.pack(pady=20)

		incorrect_password_label = tk.Label(self,text='',font=('orbitron',13),fg='White',bg='#33334d',anchor='n')
		incorrect_password_label.pack(fill='both',expand=False)

		bottom_frame = tk.Frame(self,relief='raised',borderwidth=3)
		bottom_frame.pack(fill='x',side='bottom')


if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()

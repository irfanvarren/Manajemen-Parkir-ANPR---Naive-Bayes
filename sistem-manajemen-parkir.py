import sys
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore
from PyQt5 import QtWidgets 
from PyQt5 import QtGui 
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout, QFileDialog, QTableWidgetItem, QMessageBox
from PyQt5.QtCore import QTimer, Qt, QIODevice, QLockFile
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtNetwork import QLocalSocket, QLocalServer

import os
from contextlib import redirect_stdout
import numpy as np
import cv2 as cv

from sklearn.model_selection import train_test_split
import json
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import random
from ultralytics import YOLO
import datetime
from collections import defaultdict
import threading
import multiprocessing
multiprocessing.freeze_support()


class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(860, 680)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("smk_kristen_immanuel_pontianak.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        mainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setMaximumSize(QtCore.QSize(50, 50))
        self.label_7.setText("")
        self.label_7.setPixmap(QtGui.QPixmap("smk_kristen_immanuel_pontianak.png"))
        self.label_7.setScaledContents(True)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_2.addWidget(self.label_7)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(21)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.label_5.mousePressEvent = self.open_input_citra
        
        self.verticalLayout.addWidget(self.label_5)
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout.addWidget(self.line_2)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.label_3.mousePressEvent = self.open_input_pemilik
        
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.verticalLayout.addWidget(self.line_3)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_6.setFont(font)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.verticalLayout.addWidget(self.label_6)
        self.label_6.mousePressEvent = self.open_laporan_hasil_identifikasi
        
        self.line_4 = QtWidgets.QFrame(self.centralwidget)
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.verticalLayout.addWidget(self.line_4)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.verticalLayout.addWidget(self.label_4)
        self.label_4.mousePressEvent = self.open_laporan_informasi_parkir
        
        self.gridLayout_2.addLayout(self.verticalLayout, 0, 0, 1, 1)
        mainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)
        
    
    
    def open_input_citra(self, event):
        self.second_window = InputCitraWindow(self)
        self.second_window.show()
    def open_input_pemilik(self, event):
        self.third_window = InputPemilikWindow()
        self.third_window.show()
    def open_laporan_hasil_identifikasi(self, event = None,identification_data = [], license_numbers = [], location = '', names = []):
        if(len(identification_data) == 0 ):
            try:
                with open('data-identifikasi.json', 'r') as file:
                    file_content = file.read()
                    if file_content.strip() == "":
                        identification_data= []
                    else:
                        identification_data = json.loads(file_content)
            except FileNotFoundError:
                identification_data = []
        self.fourth_window = LaporanHasilIdentifikasiWindow(identification_data, license_numbers, location, names)
        self.fourth_window.show()
    def open_laporan_informasi_parkir(self, event):
        self.fifth_window = LaporanInformasiParkirWindow()
        self.fifth_window.show()
        
    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "Sistem Manajemen Parkir SMK Kristen Immanuel"))
        self.label_4.setText(_translate("mainWindow", "Melihat laporan informasi parkir"))
        self.label_2.setText(_translate("mainWindow", "Sistem Manajemen Parkir SMK Kristen Immanuel"))
        self.label_3.setText(_translate("mainWindow", "Input data pemilik kendaraan"))
        self.label_6.setText(_translate("mainWindow", "Melihat laporan hasil identifikasi plat nomor kendaraan"))
        self.label_5.setText(_translate("mainWindow", "Input data plat nomor kendaraan"))


class Ui_InputCitraWindow(object):
    def __init__(self, parent, main_window):
        self.main_window = main_window
        self.parent = parent
        self.selected_file_path = ""
        self.location = "Tempat parkir guru"
        self.input_type = "File"
        
        def getint(name):
            basename = name.partition('.')
            return int(basename[0])
        classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        raw_labels = []    
        with open('datasets/smk-immanuel/new-sheared-rotated-labels.txt','r') as f:
            for line in f:        
                 raw_labels += str(line.strip())
        data = []
        labels =[]
        for lbl in raw_labels:
            labels.append(classes.index(lbl))


        folder_name = 'datasets/smk-immanuel/new-sheared-rotated-centered-characters'


        file_list = sorted(os.listdir(folder_name),key = getint)


        canny_features = []
        for filename in file_list:
            raw_im = cv.imread(os.path.join(folder_name,filename),0)
            (thresh, im) = cv.threshold(raw_im, 128, 255, cv.THRESH_BINARY)
            edges = cv.Canny(im,100,200)
            canny_features.append(edges)
            gray = cv.cvtColor(raw_im, cv.COLOR_GRAY2BGR)
            data.append(im)

        data = np.array(data, dtype='float32')
        data = data / 255.0

        canny_features = np.array(canny_features)
        canny_features = canny_features.reshape(data.shape[0],-1)

        flatten_data = data.reshape(data.shape[0],-1)


        combined_features = np.hstack((flatten_data, canny_features))
        X_train = combined_features
        X_test = []
        test_canny_features = []
        test_folder_name = 'datasets/smk-immanuel/test'

        test_list = sorted(os.listdir(test_folder_name),key = getint)
        for filename in test_list:
            raw_im = cv.imread(os.path.join(test_folder_name,filename),0)
            (thresh, im) = cv.threshold(raw_im, 128, 255, cv.THRESH_BINARY)
            gray = cv.cvtColor(raw_im, cv.COLOR_GRAY2BGR)
            edges = cv.Canny(im,100,200)
            test_canny_features.append(edges)
            X_test.append(im)
        X_test = np.array(X_test, dtype='float32')
        X_test = X_test.reshape(X_test.shape[0],-1)
        test_canny_features = np.array(test_canny_features)
        test_canny_features = test_canny_features.reshape(test_canny_features.shape[0],-1)
        X_test = np.hstack((X_test, test_canny_features)) 

        y_train = raw_labels
        y_test = []
        with open('datasets/smk-immanuel/test-labels.txt','r') as f:
            for line in f:        
                 y_test += str(line.strip())


        nb = MultinomialNB()

        nb.fit(X_train, y_train)
        self.nb = nb
        
    def setupUi(self, InputCitraWindow):
        InputCitraWindow.setObjectName("InputCitraWindow")
        InputCitraWindow.setWindowModality(QtCore.Qt.NonModal)
        InputCitraWindow.resize(1024, 768)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(InputCitraWindow.sizePolicy().hasHeightForWidth())
        InputCitraWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(InputCitraWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy)
        self.label_7.setMaximumSize(QtCore.QSize(50, 50))
        self.label_7.setText("")
        self.label_7.setPixmap(QtGui.QPixmap("smk_kristen_immanuel_pontianak.png"))
        self.label_7.setScaledContents(True)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout.addWidget(self.label_7)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(21)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox.sizePolicy().hasHeightForWidth())
        self.comboBox.setSizePolicy(sizePolicy)
        self.comboBox.setEditable(False)
        self.comboBox.setInsertPolicy(QtWidgets.QComboBox.InsertBeforeCurrent)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.currentIndexChanged.connect(self.start_video_capture)
        
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.comboBox)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.comboBox_2 = QtWidgets.QComboBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_2.sizePolicy().hasHeightForWidth())
        self.comboBox_2.setSizePolicy(sizePolicy)
        self.comboBox_2.setEditable(False)
        self.comboBox_2.setInsertPolicy(QtWidgets.QComboBox.InsertBeforeCurrent)
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.comboBox_2)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setStyleSheet("color : rgb(255, 85, 0)")
        self.label_8.setObjectName("label_8")
        self.label_8.setVisible(False)
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.label_8)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.process_image)
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.pushButton_2)   
        

        
        self.select_file_widget = QtWidgets.QWidget()
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.select_file_widget)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0) 
        
        # Create a push button for file selection
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setVisible(True)
        self.pushButton.clicked.connect(self.open_file_dialog)
        self.horizontalLayout_6.addWidget(self.pushButton)
        
        # Create a label for the selected file
        self.selected_file_label = QtWidgets.QLabel(self.centralwidget)
        self.selected_file_label.setObjectName("selected_file_label")
        self.selected_file_label.setVisible(True)  # Initially, hide the label
        self.horizontalLayout_6.addWidget(self.selected_file_label)

        
        
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.select_file_widget)
        
        self.horizontalLayout_5.addLayout(self.formLayout)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setUnderline(False)
        self.label_6.setFont(font)
        self.label_6.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_6.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_6.setObjectName("label_6")
        self.label_6.setVisible(False)
        self.verticalLayout_5.addWidget(self.label_6)
        self.video_capture = cv.VideoCapture()
        self.comboBox_camera = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_camera.setObjectName("comboBox_camera")
        self.comboBox_camera.setVisible(False)
        self.comboBox_camera.currentIndexChanged.connect(self.change_camera)
        self.verticalLayout_5.addWidget(self.comboBox_camera)
        self.populate_cameras()

      
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.label_5.setObjectName("label_5")
        
        self.verticalLayout_5.addWidget(self.label_5)
        
        self.horizontalLayout_5.addStretch(1)  # Add stretchable space to push the label to the right
        self.horizontalLayout_5.addLayout(self.verticalLayout_5)
    
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.verticalLayout_3.addLayout(self.verticalLayout)
        InputCitraWindow.setCentralWidget(self.centralwidget)
        
        self.loading_overlay = QWidget(self.centralwidget)
        self.loading_overlay.setStyleSheet("background-color: rgba(0, 0, 0, 128);")
        
        self.loading_overlay.setVisible(False)

        self.loading_layout = QVBoxLayout(self.loading_overlay)
        self.loading_layout.setAlignment(Qt.AlignCenter)
        self.loading_label = QLabel("Loading, please wait...")
        self.loading_label.setStyleSheet("color: white;")
        self.loading_layout.addWidget(self.loading_label)
        

        self.retranslateUi(InputCitraWindow)
        
        QtCore.QMetaObject.connectSlotsByName(InputCitraWindow)
        
        
        self.timer = QTimer(InputCitraWindow)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
    def change_camera(self, index):
        self.video_capture.release()
        camera_id = index
        self.video_capture = cv.VideoCapture(camera_id, cv.CAP_DSHOW)
    
    def populate_cameras(self):
        camera_id = 0
        while True:
            cap = cv.VideoCapture(camera_id, cv.CAP_DSHOW)
            if not cap.isOpened():
                break
            ret, frame = cap.read()
            if ret:
                self.comboBox_camera.addItem(f"Camera {camera_id}")
            cap.release()
            camera_id += 1
    def start_video_capture(self):
        selected_index = self.comboBox.currentIndex()  # Subtract 1 to account for "Select Camera" item
        self.timer.stop()
        if self.video_capture.isOpened():
            self.video_capture.release()
            self.label_5.clear()
        
        if selected_index == 0 :
            self.comboBox_camera.setVisible(False)
            self.label_6.setVisible(False)
            self.pushButton.setVisible(True)
            self.selected_file_label.setVisible(True)
            self.label_3.setVisible(True)
        elif selected_index == 1:
            self.comboBox_camera.setVisible(True)
            self.label_6.setVisible(True)
            self.pushButton.setVisible(False)
            self.selected_file_label.setVisible(False)
            self.label_3.setVisible(False)
            self.video_capture.open(0)
            self.timer.start(33)

        
    def update_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            max_width = 640
            max_height = 480
            pixmap = pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio)
            self.label_5.setPixmap(pixmap)
            self.last_frame = frame.copy()
    def open_file_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        
        file_dialog = QFileDialog()
        file_dialog.setDirectory(QtCore.QDir.currentPath())
        file_dialog.setNameFilter("Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
        file_dialog.setViewMode(QFileDialog.Detail)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                file_path = selected_files[0]
                self.selected_file_path = file_path
                self.selected_file_label.setText(f'Selected file: {file_path}')
            
    def closeEvent(self, event):
        self.timer.stop()
        if self.video_capture.isOpened():
            self.video_capture.release()
        event.accept()
        
   
    def thread_safe_predict(self,model, image):
            # Instantiate a new model inside the thread
            results = model.predict(image,  conf=0.5, verbose = False)  # save plotted images

            # Store the result in the thread object
            setattr(threading.current_thread(), 'results', results)
    def thread_safe_predict_character(self,model_character, image):
            # Instantiate a new model inside the thread
            character_results = model_character.predict(image,  conf=0.15, verbose = False)# save plotted images

            # Store the result in the thread object
            setattr(threading.current_thread(), 'character_results', character_results)
    
    
    def process_image(self):
        def luminosity_grayscale(image):
            img_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            R, G, B = img_rgb[:,:,0], img_rgb[:,:,1],img_rgb[:,:,2]   
            Z = (0.2126 * R + 0.7152 * G + 0.0722 * B).astype(np.uint8)
            return Z

        def median_blur(grayscale_image):
            median = cv.medianBlur(grayscale_image, 5)
            return median

        def tophat(noise_removed_image):
            tophat = cv.morphologyEx(noise_removed_image,cv.MORPH_TOPHAT,cv.getStructuringElement(cv.MORPH_RECT, (31,31)))
            return tophat

        def otsu_threshold(contrast_enhanced_image):
            #thres, binary = cv.threshold(contrast_enhanced_image,80,255,cv.THRESH_BINARY)
            thres, binary = cv.threshold(contrast_enhanced_image,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
            return binary

        def opening(binary_image):
            opening = cv.morphologyEx(binary_image, cv.MORPH_OPEN, np.ones((3,3),np.uint8))
            return opening

        def closing(binary_image):
            closing = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, np.ones((3,3),np.uint8))
            return closing


        def create_projection(image, method) -> list:
            index1 = 1 if method == 'v' else 0
            index2 = abs(1-index1) 
            hist = np.zeros(image.shape[index1])
            for y in range(image.shape[index1]):
                for x in range(image.shape[index2]):
                    temp = image[x,y] if method == 'v' else image[y,x]
                    hist[y] += temp
            hist /= image.shape[index2]
            return hist

        def create_lines(hist, threshold) -> list:
            lines = []
            gap = []
            for i, val in enumerate(hist):
                if val <= threshold:
                    gap.append(i)
                else:
                    if gap:
                        avg = sum(gap)//len(gap)
                        lines.append(avg)
                        gap = []
            if gap:
                avg = sum(gap)//len(gap)
                lines.append(avg)
            return lines

        def split_image(image, lines, method):
            img = []

            try:
                if(len(lines) > 1):
                    for i,l in enumerate(lines):
                        pos1 = lines[i]
                        pos2 = lines[i+1]
                        temp = image[pos1: pos2] if method =='h' else image[:,pos1:pos2]
                        img.append(temp)
                elif(len(lines) == 1) : 
                    temp = image[0:lines[0]]
                    img.append(temp)
                elif(len(lines) == 0) :
                    img.append(image)
            except IndexError:
                    pass
            return img 
        def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
            lw = max(round(sum(image.shape) / 2 * 0.003), 2)
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv.rectangle(image, p1, p2, color, thickness=lw, lineType=cv.LINE_AA)
            if label:
                tf = max(lw - 1, 1)  # font thickness
                w, h = cv.getTextSize(label, 0, fontScale=lw / 4, thickness=tf)[0]  # text width, height
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3

                cv.rectangle(image, p1, p2, color, -1, cv.LINE_AA)  # filled
                cv.putText(image,
                            label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                            0,
                            lw / 4,
                            txt_color,
                            thickness=tf,
                            lineType=cv.LINE_AA)

        def plot_bboxes(image, box, labels=[], colors=[], score=True, conf=None):
            #Define COCO Labels
            if len(labels) == 0:
                labels = {0:'???'}
                label = labels[0]
            else:
                label = labels[0]
            #Define colors
            if colors == []:
            #colors = [(6, 112, 83), (253, 246, 160), (40, 132, 70), (205, 97, 162), (149, 196, 30), (106, 19, 161), (127, 175, 225), (115, 133, 176), (83, 156, 8), (182, 29, 77), (180, 11, 251), (31, 12, 123), (23, 6, 115), (167, 34, 31), (176, 216, 69), (110, 229, 222), (72, 183, 159), (90, 168, 209), (195, 4, 209), (135, 236, 21), (62, 209, 199), (87, 1, 70), (75, 40, 168), (121, 90, 126), (11, 86, 86), (40, 218, 53), (234, 76, 20), (129, 174, 192), (13, 18, 254), (45, 183, 149), (77, 234, 120), (182, 83, 207), (172, 138, 252), (201, 7, 159), (147, 240, 17), (134, 19, 233), (202, 61, 206), (177, 253, 26), (10, 139, 17), (130, 148, 106), (174, 197, 128), (106, 59, 168), (124, 180, 83), (78, 169, 4), (26, 79, 176), (185, 149, 150), (165, 253, 206), (220, 87, 0), (72, 22, 226), (64, 174, 4), (245, 131, 96), (35, 217, 142), (89, 86, 32), (80, 56, 196), (222, 136, 159), (145, 6, 219), (143, 132, 162), (175, 97, 221), (72, 3, 79), (196, 184, 237), (18, 210, 116), (8, 185, 81), (99, 181, 254), (9, 127, 123), (140, 94, 215), (39, 229, 121), (230, 51, 96), (84, 225, 33), (218, 202, 139), (129, 223, 182), (167, 46, 157), (15, 252, 5), (128, 103, 203), (197, 223, 199), (19, 238, 181), (64, 142, 167), (12, 203, 242), (69, 21, 41), (177, 184, 2), (35, 97, 56), (241, 22, 161)]
                colors = [(89, 161, 197),(67, 161, 255),(19, 222, 24),(186, 55, 2),(167, 146, 11),(190, 76, 98),(130, 172, 179),(115, 209, 128),(204, 79, 135),(136, 126, 185),(209, 213, 45),(44, 52, 10),(101, 158, 121),(179, 124, 12),(25, 33, 189),(45, 115, 11),(73, 197, 184),(62, 225, 221),(32, 46, 52),(20, 165, 16),(54, 15, 57),(12, 150, 9),(10, 46, 99),(94, 89, 46),(48, 37, 106),(42, 10, 96),(7, 164, 128),(98, 213, 120),(40, 5, 219),(54, 25, 150),(251, 74, 172),(0, 236, 196),(21, 104, 190),(226, 74, 232),(120, 67, 25),(191, 106, 197),(8, 15, 134),(21, 2, 1),(142, 63, 109),(133, 148, 146),(187, 77, 253),(155, 22, 122),(218, 130, 77),(164, 102, 79),(43, 152, 125),(185, 124, 151),(95, 159, 238),(128, 89, 85),(228, 6, 60),(6, 41, 210),(11, 1, 133),(30, 96, 58),(230, 136, 109),(126, 45, 174),(164, 63, 165),(32, 111, 29),(232, 40, 70),(55, 31, 198),(148, 211, 129),(10, 186, 211),(181, 201, 94),(55, 35, 92),(129, 140, 233),(70, 250, 116),(61, 209, 152),(216, 21, 138),(100, 0, 176),(3, 42, 70),(151, 13, 44),(216, 102, 88),(125, 216, 93),(171, 236, 47),(253, 127, 103),(205, 137, 244),(193, 137, 224),(36, 152, 214),(17, 50, 238),(154, 165, 67),(114, 129, 60),(119, 24, 48),(73, 8, 110)]
            color = random.choice(colors)
            box_label(image,box,label,color)

        
            
        self.loading_overlay.setGeometry(self.centralwidget.geometry())    
        self.loading_overlay.setVisible(True)

        model = YOLO('best.pt')
        model_character = YOLO('best-character-2.pt')
        
        selected_index = self.comboBox.currentIndex()  # Subtract 1 to account for "Select Camera" item
        results = []
        if selected_index == 0 :
            if self.selected_file_path :
                source = self.selected_file_path
                #img_path = os.path.join(folder,filename)
                #img= cv.imread(img_path)
                img = cv.imread(source)
                with open(os.devnull, 'w') as devnull:
                    with redirect_stdout(devnull):
                        yolo_thread = threading.Thread(target=self.thread_safe_predict, args=(model,source))
                        yolo_thread.start()
                        yolo_thread.join()
                        results = getattr(yolo_thread, 'results', [])
                
            else :
                print('check error')
        elif selected_index == 1:
            last_frame = self.last_frame.copy() if self.last_frame is not None else None
            if(last_frame.any()):
                img = last_frame
                with open(os.devnull, 'w') as devnull:
                    with redirect_stdout(devnull):
                        yolo_thread = threading.Thread(target=self.thread_safe_predict, args=(model,img))
                        yolo_thread.start()
                        yolo_thread.join()
                        results = getattr(yolo_thread, 'results', [])
                
       

        license_numbers = []
        for result in results:

            for result_box in result.boxes.xyxy:

                x = int(result_box[0])
                y = int(result_box[1])
                x2 = int(result_box[2])
                y2 = int(result_box[3])
                box = [x,y,x2,y2]

                plate_img =img[y:y2,x:x2]
                character_results = []
                
                yolo_thread2 = threading.Thread(target=self.thread_safe_predict_character, args=(model_character,plate_img))
                yolo_thread2.start()
                yolo_thread2.join()
                character_results = getattr(yolo_thread2, 'character_results', [])
                predicted_characters = []
                for result in character_results : 
                    if(len(result.boxes.xyxy) > 0):
                        max_y = max(row[1] for row in result.boxes.xyxy)
                        min_y = min(row[1] for row in result.boxes.xyxy)
                        center_y = (max_y + min_y) / 2
                        upper_words = [f.tolist() for f in result.boxes.xyxy if f[1] < center_y]
                        upper_words = sorted(upper_words, key=lambda x: x[0])
                        for index,result_box in enumerate(upper_words):
                            ch_x = int(result_box[0])
                            ch_y = int(result_box[1])
                            ch_x2 = int(result_box[2])
                            ch_y2 = int(result_box[3])
                            ch_box = [ch_x,ch_y,ch_x2,ch_y2]
                            label = result.names[int(result.boxes.cls[index])]


                            crop_img =plate_img[ch_y:ch_y2, ch_x:ch_x2]
                            grayscale = luminosity_grayscale(crop_img)
                            noise_remov = median_blur(grayscale)
                            contrast_enhance = tophat(noise_remov)
                            binarization = otsu_threshold(contrast_enhance)
                            horizontal_hist = create_projection(binarization, method='h')
                            horizontal_threshold = (np.max(horizontal_hist) - np.min(horizontal_hist))

                            horizontal_lines = create_lines(horizontal_hist, 0)
                            if(len(horizontal_lines) > 0):
                                if(horizontal_lines[0] > horizontal_threshold):
                                    horizontal_lines = np.insert(horizontal_lines,0,0)
                                if(len(horizontal_lines) == 1):
                                    horizontal_lines.append(int(binarization.shape[0]))
                            else:
                                horizontal_lines.append(int(0))
                                horizontal_lines.append(int(binarization.shape[0]))

                            splited_horizontal_img = split_image(binarization, horizontal_lines, method='h')
                            splited_horizontal_img = sorted(splited_horizontal_img, key=len,reverse=True)
                            horizontal_segmented_img = splited_horizontal_img[0]

                            vertical_hist = create_projection(horizontal_segmented_img, method='v')

                            vertical_threshold = (np.max(vertical_hist) - np.min(vertical_hist))
                            vertical_lines = create_lines(vertical_hist, 0)

                            if(len(vertical_lines) > 0):
                                if(vertical_lines[0] != 0):
                                    vertical_lines = np.insert(vertical_lines,0,0)
                                if(vertical_lines[0] > vertical_threshold):
                                    vertical_lines = np.insert(vertical_lines,0,0)
                                if(len(vertical_lines) == 1):
                                    vertical_lines.append(int(horizontal_segmented_img.shape[1]))                
                                if(vertical_lines[-1] != int(horizontal_segmented_img.shape[1])):
                                    vertical_lines = np.append(vertical_lines,int(horizontal_segmented_img.shape[1]))
                            else:
                                vertical_lines.append(int(0))
                                vertical_lines.append(int(img.shape[0]))


                            splited_vertical_img = split_image(horizontal_segmented_img, vertical_lines, method='v')
                            splited_vertical_img = sorted(splited_vertical_img, key=lambda s: s.size,reverse=True)

                            vertical_segmented_img = splited_vertical_img[0]

                            vertical_x,vertical_y = np.where(vertical_segmented_img != 0)
                            if(len(vertical_x) > 0 and len(vertical_y) > 0):
                                min_x, max_x = min(vertical_x),max(vertical_x)
                                min_y, max_y = min(vertical_y),max(vertical_y)
                                character_img = vertical_segmented_img[min_x:max_x,min_y:max_y]
                                if(len(character_img) > 0 and character_img.size > 0):
                                    character_h,character_w = character_img.shape[:2] # old_size is in (height, width) format
                                    if(character_h >  161):
                                        character_w = int((character_w / character_h) * 161)
                                        character_h = 161
                                        character_img = cv.resize(character_img, (character_w,161))
                                    delta_w = 94- character_w
                                    delta_h = 161- character_h
                                    top, bottom = delta_h//2, delta_h-(delta_h//2)
                                    left, right = delta_w//2, delta_w-(delta_w//2)

                                    color = [0, 0, 0]
                                    centered_character = cv.copyMakeBorder(character_img, top, bottom, left, right, None,value=color)
                                    edges = cv.Canny(centered_character,100,200)

                                    character_features =  np.hstack((centered_character.reshape(1,centered_character.size), edges.reshape(1, centered_character.size)))
                                    predicted_character = self.nb.predict(character_features)
                                    predicted_characters.append(predicted_character)
                                    plot_bboxes(plate_img,ch_box,predicted_character)
                if(len(predicted_characters) > 0):
                    license_numbers.append(''.join(np.concatenate(predicted_characters)))


        QTimer.singleShot(1000, lambda:self.save_data(license_numbers))
            
            
            
        
       
        
    def save_data(self,license_numbers):
        input_type = self.comboBox.currentText()
        location = self.comboBox_2.currentText()
        self.location = location
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_data = [license_numbers, input_type, location, current_time]
  
        with open('data-identifikasi.json', 'r+') as file:
            data = json.load(file) if os.stat('data-identifikasi.json').st_size != 0 else []
            data.append(new_data)
            file.seek(0)
            json.dump(data, file, indent=2)
            
        self.loading_overlay.setVisible(False)
        self.main_window.open_laporan_hasil_identifikasi(None,data,license_numbers,self.location,[])
            
    def retranslateUi(self, InputCitraWindow):
        _translate = QtCore.QCoreApplication.translate
        InputCitraWindow.setWindowTitle(_translate("InputCitraWindow", "Sistem Manajemen Parkir SMK Kristen Immanuel"))
        self.label_2.setText(_translate("InputCitraWindow", "Sistem Manajemen Parkir SMK Kristen Immanuel"))
        self.label.setText(_translate("InputCitraWindow", "Jenis Input"))
        self.comboBox.setCurrentText(_translate("InputCitraWindow", "File"))
        self.comboBox.setItemText(0, _translate("InputCitraWindow", "File"))
        self.comboBox.setItemText(1, _translate("InputCitraWindow", "Kamera 1"))
        self.label_3.setText(_translate("InputCitraWindow", "File"))
        self.comboBox_2.setCurrentText(_translate("InputCitraWindow", "Tempat parkir guru"))
        self.comboBox_2.setItemText(0, _translate("InputCitraWindow", "Tempat parkir guru"))
        self.comboBox_2.setItemText(1, _translate("InputCitraWindow", "Tempat parkir depan"))
        self.comboBox_2.setItemText(2, _translate("InputCitraWindow", "Tempat parkir belakang"))
        self.label_4.setText(_translate("InputCitraWindow", "Lokasi parkir"))
        self.label_8.setText(_translate("InputCitraWindow", "Data yang diinput tidak lengkap !"))
        self.pushButton_2.setText(_translate("InputCitraWindow", "Proses"))
        self.pushButton.setText(_translate("InputCitraWindow", "Pilih File"))
        self.label_6.setText(_translate("InputCitraWindow", "Kamera : "))
        #self.label_5.setText(_translate("InputCitraWindow", "TextLabel"))

class Ui_InputPemilikWindow(object):
    def setupUi(self, InputPemilikWindow):
        InputPemilikWindow.setObjectName("InputPemilikWindow")
        InputPemilikWindow.resize(969, 520)
        self.centralwidget = QtWidgets.QWidget(InputPemilikWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setMaximumSize(QtCore.QSize(50, 50))
        self.label_7.setText("")
        self.label_7.setPixmap(QtGui.QPixmap("smk_kristen_immanuel_pontianak.png"))
        self.label_7.setScaledContents(True)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout.addWidget(self.label_7)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(21)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        
        
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.formLayout.setObjectName("formLayout")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label)
        self.plate_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.plate_edit.setObjectName("lineEdit_2")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.plate_edit)
        self.name_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.name_edit.setObjectName("lineEdit")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.name_edit)
        self.nis_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.nis_edit.setObjectName("lineEdit_3")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.nis_edit)
        
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.save_data)
        self.horizontalLayout_2.addWidget(self.pushButton_2)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.update_data)
        self.horizontalLayout_2.addWidget(self.pushButton)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(self.delete_data)
        self.horizontalLayout_2.addWidget(self.pushButton_3)
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setStyleSheet("color : rgb(255, 85, 0)")
        self.label_8.setObjectName("label_8")
        self.label_8.setVisible(False)
        self.horizontalLayout_2.addWidget(self.label_8)
        self.formLayout.setLayout(3, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_2)
        self.verticalLayout_3.addLayout(self.formLayout)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setMinimumSize(QtCore.QSize(949, 203))
        self.tableWidget.setAlternatingRowColors(True)
        self.tableWidget.setShowGrid(True)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(3)
        self.tableWidget.clicked.connect(self.fill_form)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
       
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 3, item)
        self.tableWidget.horizontalHeader().setCascadingSectionResizes(False)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(139)
        self.tableWidget.horizontalHeader().setMinimumSectionSize(44)
        self.tableWidget.horizontalHeader().setSortIndicatorShown(False)
        self.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.tableWidget.verticalHeader().setVisible(True)
        self.tableWidget.verticalHeader().setSortIndicatorShown(False)
        self.tableWidget.verticalHeader().setStretchLastSection(False)
        self.verticalLayout.addWidget(self.tableWidget)
        self.verticalLayout_3.addLayout(self.verticalLayout)
        self.verticalLayout_3.setStretch(2, 1)
        InputPemilikWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(InputPemilikWindow)
        QtCore.QMetaObject.connectSlotsByName(InputPemilikWindow)
      
        # Initialize table data
        self.table_data = []
        self.load_data_from_json()

        # Display data in table
        self.display_data_in_table()
        
    def save_data(self):
        plate_number = self.plate_edit.text()
        name = self.name_edit.text()
        nis = self.nis_edit.text()

        if plate_number and name and nis:
            new_row = {plate_number: {'name': name, 'nis': nis}}
            self.table_data.update(new_row)
            self.display_data_in_table()
            self.clear_input_fields()
            self.save_data_to_json()
        else:
            self.show_error_message("Please fill in all fields.")

    def update_data(self):
        selected_plate_number = self.tableWidget.item(self.tableWidget.currentRow(), 0).text()
        if selected_plate_number:
            name = self.name_edit.text()
            nis = self.nis_edit.text()

            if name and nis:
                self.table_data[selected_plate_number]['name'] = name
                self.table_data[selected_plate_number]['nis'] = nis
                self.display_data_in_table()
                self.clear_input_fields()
                self.save_data_to_json()
            else:
                self.show_error_message("Please fill in all fields.")
        else:
            self.show_error_message("Select a row to update.")

    def delete_data(self):
        selected_plate_number = self.tableWidget.item(self.tableWidget.currentRow(), 0).text()
        if selected_plate_number:
            del self.table_data[selected_plate_number]
            self.display_data_in_table()
            self.clear_input_fields()
            self.save_data_to_json()
        else:
            self.show_error_message("Select a row to delete.")
    def fill_form(self, item):
        row = item.row()
        if row >= 0:
            plate_number = self.tableWidget.item(row, 0).text()
            name = self.tableWidget.item(row, 1).text()
            nis = self.tableWidget.item(row, 2).text()
            self.plate_edit.setText(plate_number)
            self.name_edit.setText(name)
            self.nis_edit.setText(nis)
            
    def display_data_in_table(self):
        self.tableWidget.setRowCount(0)
        for row_num, (plate_number, data) in enumerate(self.table_data.items()):
            self.tableWidget.insertRow(row_num)
            self.tableWidget.setItem(row_num, 0, QTableWidgetItem(plate_number))
            self.tableWidget.setItem(row_num, 1, QTableWidgetItem(data['name']))
            self.tableWidget.setItem(row_num, 2, QTableWidgetItem(data['nis']))
    
    def load_data_from_json(self):
        try:
            with open('data-pemilik.json', 'r') as file:
                file_content = file.read()
                if file_content.strip() == "":
                    self.table_data = {}
                else:
                    self.table_data = json.loads(file_content)
        except FileNotFoundError:
            self.table_data = {}

    def save_data_to_json(self):
        with open('data-pemilik.json', 'w') as file:
            json.dump(self.table_data, file, indent=4)
                
    def clear_input_fields(self):
        self.plate_edit.clear()
        self.name_edit.clear()
        self.nis_edit.clear()

    def show_error_message(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(message)
        msg_box.setWindowTitle("Error")
        msg_box.exec_()


    def retranslateUi(self, InputPemilikWindow):
        _translate = QtCore.QCoreApplication.translate
        InputPemilikWindow.setWindowTitle(_translate("InputPemilikWindow", "InputPemilikWindow"))
        self.label_2.setText(_translate("InputPemilikWindow", "Sistem Manajemen Parkir SMK Kristen Immanuel"))
       
        self.label.setText(_translate("InputPemilikWindow", "Nomor Induk Siswa"))
        self.label_3.setText(_translate("InputPemilikWindow", "Nomor Plat Kendaraan"))
        self.label_4.setText(_translate("InputPemilikWindow", "Nama"))
        self.pushButton_2.setText(_translate("InputPemilikWindow", "Simpan"))
        self.pushButton.setText(_translate("InputPemilikWindow", "Ubah"))
        self.pushButton_3.setText(_translate("InputPemilikWindow", "Hapus"))
        self.label_8.setText(_translate("InputPemilikWindow", "Data yang diinput tidak lengkap !"))
        self.tableWidget.setSortingEnabled(False)
        
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("InputPemilikWindow", "Nomor Plat Kendaraan"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("InputPemilikWindow", "Nama"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("InputPemilikWindow", "Nomor Induk Siswa"))
        __sortingEnabled = self.tableWidget.isSortingEnabled()
        self.tableWidget.setSortingEnabled(False)
        self.tableWidget.setSortingEnabled(__sortingEnabled)

class Ui_LaporanHasilIdentifikasiWindow(object):
    def __init__(self,data, plate_number,location,names):
        self.plate_number = plate_number
        self.location = location
        self.data_pemilik = []
        self.data = data
        try:
            with open('data-pemilik.json', 'r') as file:
                file_content = file.read()
                if file_content.strip() == "":
                    self.data_pemilik =  {}
                else:
                    self.data_pemilik = json.loads(file_content)
        except FileNotFoundError:
            self.data_pemilik = {}
        names = [self.data_pemilik[item]['name'] if item in self.data_pemilik else '-' for item in self.plate_number]
        self.names = names
        
        
        
    def setupUi(self, LaporanHasilIdentifikasiWindow):
        LaporanHasilIdentifikasiWindow.setObjectName("LaporanHasilIdentifikasiWindow")
        LaporanHasilIdentifikasiWindow.resize(1033, 683)
        self.centralwidget = QtWidgets.QWidget(LaporanHasilIdentifikasiWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setSpacing(12)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setMaximumSize(QtCore.QSize(50, 50))
        self.label_7.setText("")
        self.label_7.setPixmap(QtGui.QPixmap("smk_kristen_immanuel_pontianak.png"))
        self.label_7.setScaledContents(True)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout.addWidget(self.label_7)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(21)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.formLayout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setRowWrapPolicy(QtWidgets.QFormLayout.DontWrapRows)
        self.formLayout.setLabelAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.formLayout.setFormAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.formLayout.setVerticalSpacing(12)
        self.formLayout.setObjectName("formLayout")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.label_5)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.label_8)
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.label_3)
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_9)
        self.verticalLayout_2.addLayout(self.formLayout)
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setMinimumSize(QtCore.QSize(949, 203))
        self.tableWidget.setAlternatingRowColors(True)
        self.tableWidget.setShowGrid(True)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setRowCount(10)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(7, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(8, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(9, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
     
        self.tableWidget.setItem(0, 1, item)
        self.tableWidget.horizontalHeader().setCascadingSectionResizes(False)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(201)
        self.tableWidget.horizontalHeader().setMinimumSectionSize(44)
        self.tableWidget.horizontalHeader().setSortIndicatorShown(False)
        self.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.tableWidget.verticalHeader().setVisible(True)
        self.tableWidget.verticalHeader().setSortIndicatorShown(False)
        self.tableWidget.verticalHeader().setStretchLastSection(False)
        self.verticalLayout_2.addWidget(self.tableWidget)
        self.verticalLayout_2.setStretch(2, 1)
        LaporanHasilIdentifikasiWindow.setCentralWidget(self.centralwidget)
        
        self.load_data_from_json()
        self.retranslateUi(LaporanHasilIdentifikasiWindow)
        QtCore.QMetaObject.connectSlotsByName(LaporanHasilIdentifikasiWindow)
    
    def load_data_from_json(self):
        row_index = 0
        
        for row in self.data[::-1]:
            
            numbers = row[0] if isinstance(row[0], list) else [row[0]]
            for number in numbers:
                self.tableWidget.setItem(row_index, 0, QTableWidgetItem(number))
                nama = self.data_pemilik.get(number, {}).get('name', '-')
                self.tableWidget.setItem(row_index, 1, QTableWidgetItem(nama))
                lokasi = row[2]
                self.tableWidget.setItem(row_index, 2, QTableWidgetItem(lokasi))
                row_index += 1
              
                              
    def retranslateUi(self, LaporanHasilIdentifikasiWindow):
        _translate = QtCore.QCoreApplication.translate
        LaporanHasilIdentifikasiWindow.setWindowTitle(_translate("LaporanHasilIdentifikasiWindow", "LaporanHasilIdentifikasiWindow"))
        self.label_2.setText(_translate("LaporanHasilIdentifikasiWindow", "Sistem Manajemen Parkir SMK Kristen Immanuel"))
        self.label_4.setText(_translate("LaporanHasilIdentifikasiWindow", "Nama Pemilik Kendaraan"))
        
        if(self.names is not None and len(self.names) > 0):
            self.label_5.setText(_translate("LaporanHasilIdentifikasiWindow", ', '.join(self.names)))
        else :
            self.label_5.setText(_translate("LaporanHasilIdentifikasiWindow", "-"))
        
        self.label_6.setText(_translate("LaporanHasilIdentifikasiWindow", "Lokasi Tempat Parkir"))
        
        if(self.location is not None and len(self.location) > 0):
            self.label_8.setText(_translate("LaporanHasilIdentifikasiWindow", self.location))
        else :
            self.label_8.setText(_translate("LaporanHasilIdentifikasiWindow", "-"))
            
        self.label.setText(_translate("LaporanHasilIdentifikasiWindow", "Nomor Plat Kendaraan"))
        
        if(self.plate_number is not None and len(self.plate_number) > 0):
            self.label_3.setText(_translate("LaporanHasilIdentifikasiWindow", ', '.join(self.plate_number)))
        else :
            self.label_3.setText(_translate("LaporanHasilIdentifikasiWindow", "-"))
            
        self.label_9.setText(_translate("LaporanHasilIdentifikasiWindow", "Laporan Hasil Identifikasi"))
        self.tableWidget.setSortingEnabled(False)
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("LaporanHasilIdentifikasiWindow", "Nomor Plat Kendaraan"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("LaporanHasilIdentifikasiWindow", "Nama"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("LaporanHasilIdentifikasiWindow", "Lokasi Tempat Parkir"))
        __sortingEnabled = self.tableWidget.isSortingEnabled()
        self.tableWidget.setSortingEnabled(False)
        self.tableWidget.setSortingEnabled(__sortingEnabled)


class Ui_LaporanInformasiParkirWindow(object):
    def setupUi(self, LaporanInformasiParkirWindow):
        LaporanInformasiParkirWindow.setObjectName("LaporanInformasiParkirWindow")
        LaporanInformasiParkirWindow.resize(1036, 750)
        self.centralwidget = QtWidgets.QWidget(LaporanInformasiParkirWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setSpacing(12)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setMaximumSize(QtCore.QSize(50, 50))
        self.label_7.setText("")
        self.label_7.setPixmap(QtGui.QPixmap("smk_kristen_immanuel_pontianak.png"))
        self.label_7.setScaledContents(True)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout.addWidget(self.label_7)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(21)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_18 = QtWidgets.QLabel(self.centralwidget)
        self.label_18.setMinimumSize(QtCore.QSize(0, 40))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(False)
        font.setWeight(50)
        self.label_18.setFont(font)
        self.label_18.setObjectName("label_18")
        self.verticalLayout_3.addWidget(self.label_18)
        self.label_19 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_19.setFont(font)
        self.label_19.setObjectName("label_19")
        self.verticalLayout_3.addWidget(self.label_19)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setContentsMargins(-1, 0, -1, -1)
        self.horizontalLayout_2.setSpacing(6)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.formLayout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setRowWrapPolicy(QtWidgets.QFormLayout.DontWrapRows)
        self.formLayout.setLabelAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.formLayout.setFormAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.formLayout.setVerticalSpacing(12)
        self.formLayout.setObjectName("formLayout")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy)
        self.label_9.setMinimumSize(QtCore.QSize(0, 19))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_9.setObjectName("label_9")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.SpanningRole, self.label_9)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setMinimumSize(QtCore.QSize(240, 0))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.label_5)
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.label_8)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setMinimumSize(QtCore.QSize(200, 0))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.horizontalLayout_2.addLayout(self.formLayout)
        self.formLayout_3 = QtWidgets.QFormLayout()
        self.formLayout_3.setLabelAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.formLayout_3.setFormAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.formLayout_3.setVerticalSpacing(12)
        self.formLayout_3.setObjectName("formLayout_3")
        self.label_17 = QtWidgets.QLabel(self.centralwidget)
        self.label_17.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_17.setFont(font)
        self.label_17.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_17.setObjectName("label_17")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_17)
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_16)
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.label_11)
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.formLayout_3.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_15)
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.formLayout_3.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.label_12)
        self.horizontalLayout_2.addLayout(self.formLayout_3)
        self.formLayout_2 = QtWidgets.QFormLayout()
        self.formLayout_2.setVerticalSpacing(12)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_10.setFont(font)
        self.label_10.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_10.setObjectName("label_10")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_10)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.label)
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_13)
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.label_14)
        self.horizontalLayout_2.addLayout(self.formLayout_2)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.verticalLayout_2.addLayout(self.verticalLayout_3)
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setMinimumSize(QtCore.QSize(949, 203))
        self.tableWidget.setAlternatingRowColors(True)
        self.tableWidget.setShowGrid(True)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setRowCount(10)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(7, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(8, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(9, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 1, item)
        self.tableWidget.horizontalHeader().setCascadingSectionResizes(False)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(201)
        self.tableWidget.horizontalHeader().setMinimumSectionSize(44)
        self.tableWidget.horizontalHeader().setSortIndicatorShown(False)
        self.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.tableWidget.verticalHeader().setVisible(True)
        self.tableWidget.verticalHeader().setSortIndicatorShown(False)
        self.tableWidget.verticalHeader().setStretchLastSection(False)
        self.verticalLayout_2.addWidget(self.tableWidget)
        self.verticalLayout_2.setStretch(2, 1)
        LaporanInformasiParkirWindow.setCentralWidget(self.centralwidget)
        self.load_data_from_json()
        self.retranslateUi(LaporanInformasiParkirWindow)
        QtCore.QMetaObject.connectSlotsByName(LaporanInformasiParkirWindow)
    
    def load_data_from_json(self):
        
        try:
            with open('data-identifikasi.json', 'r') as file:
                file_content = file.read()
                if file_content.strip() == "":
                    identification_data= []
                else:
                    identification_data = json.loads(file_content)
        except FileNotFoundError:
            identification_data = []
        self.data = identification_data
        row_index = 0
        for row in self.data[::-1]:
            numbers = row[0] if isinstance(row[0], list) else [row[0]]
            for number in numbers:
                self.tableWidget.setItem(row_index, 0, QTableWidgetItem(number))
                lokasi = row[2]
                self.tableWidget.setItem(row_index, 1, QTableWidgetItem(lokasi))
                waktu = row[3]
                self.tableWidget.setItem(row_index, 2, QTableWidgetItem(waktu))
                row_index += 1

    def retranslateUi(self, LaporanInformasiParkirWindow):
        _translate = QtCore.QCoreApplication.translate
        LaporanInformasiParkirWindow.setWindowTitle(_translate("LaporanInformasiParkirWindow", "LaporanInformasiParkirWindow"))
        self.label_2.setText(_translate("LaporanInformasiParkirWindow", "Sistem Manajemen Parkir SMK Kristen Immanuel"))
        self.label_18.setText(_translate("LaporanInformasiParkirWindow", "Informasi Parkir"))
        self.label_19.setText(_translate("LaporanInformasiParkirWindow", "Lokasi Tempat Parkir :"))
        self.label_9.setText(_translate("LaporanInformasiParkirWindow", "Tempat Parkir Guru"))
        self.label_4.setText(_translate("LaporanInformasiParkirWindow", "Jumlah kendaraan "))
        
        
        self.label_6.setText(_translate("LaporanInformasiParkirWindow", "Jumlah Tempat yang masih tersedia "))
        self.label_17.setText(_translate("LaporanInformasiParkirWindow", "Tempat Parkir Depan"))
        self.label_16.setText(_translate("LaporanInformasiParkirWindow", "Jumlah kendaraan"))
        
        self.label_15.setText(_translate("LaporanInformasiParkirWindow", "Jumlah Tempat yang masih tersedia"))
        
        self.label_10.setText(_translate("LaporanInformasiParkirWindow", "Tempat Parkir Belakang"))
        self.label_3.setText(_translate("LaporanInformasiParkirWindow", "Jumlah kendaraan"))
        
        self.label_13.setText(_translate("LaporanInformasiParkirWindow", "Jumlah Tempat yang masih tersedia"))
        
        
        
       

        try:
            with open('data-identifikasi.json', 'r') as file:
                file_content = file.read()
                if file_content.strip() == "":
                    data= []
                else:
                    data = json.loads(file_content)
        except FileNotFoundError:
            data = []

        # Get today's date
        today = datetime.datetime.now().strftime("%Y-%m-%d")

        # Filter data for today's date and index 1 is "File"
        filtered_data = [item for item in data if item[3].startswith(today) and item[1] == "File"]

        # Group data based on index 2
        grouped_data = defaultdict(list)
        for item in filtered_data:
            grouped_data[item[2]].append(item[0] if isinstance(item[0], str) else tuple(item[0]))

        # Count elements in index 0 within each group
        group_counts = {"Tempat parkir guru" : [], "Tempat parkir depan" : [], "Tempat parkir belakang" : []}
        for key, group in grouped_data.items():
            counts = defaultdict(int)
            for item in group:
                if isinstance(item, tuple):
                    for sub_item in item:
                        counts[sub_item] += 1
                else:
                    counts[item] += 1
            group_counts[key] = counts

        self.label_5.setText(_translate("LaporanInformasiParkirWindow", ": " + str(len(group_counts["Tempat parkir guru"]))))
        self.label_8.setText(_translate("LaporanInformasiParkirWindow", ": " + str((25- len(group_counts["Tempat parkir guru"])))))
        
        self.label_11.setText(_translate("LaporanInformasiParkirWindow", ": " + str(len(group_counts["Tempat parkir depan"]))))
        self.label_12.setText(_translate("LaporanInformasiParkirWindow", ": " + str((75- len(group_counts["Tempat parkir depan"])))))
        
        self.label.setText(_translate("LaporanInformasiParkirWindow", ": " + str(len(group_counts["Tempat parkir belakang"]))))
        self.label_14.setText(_translate("LaporanInformasiParkirWindow", ": " + str((200- len(group_counts["Tempat parkir belakang"])))))
        
        self.tableWidget.setSortingEnabled(False)
        item = self.tableWidget.verticalHeaderItem(0)
        item.setText(_translate("LaporanInformasiParkirWindow", "1"))
        item = self.tableWidget.verticalHeaderItem(1)
        item.setText(_translate("LaporanInformasiParkirWindow", "2"))
        item = self.tableWidget.verticalHeaderItem(2)
        item.setText(_translate("LaporanInformasiParkirWindow", "3"))
        item = self.tableWidget.verticalHeaderItem(3)
        item.setText(_translate("LaporanInformasiParkirWindow", "4"))
        item = self.tableWidget.verticalHeaderItem(4)
        item.setText(_translate("LaporanInformasiParkirWindow", "5"))
        item = self.tableWidget.verticalHeaderItem(5)
        item.setText(_translate("LaporanInformasiParkirWindow", "6"))
        item = self.tableWidget.verticalHeaderItem(6)
        item.setText(_translate("LaporanInformasiParkirWindow", "7"))
        item = self.tableWidget.verticalHeaderItem(7)
        item.setText(_translate("LaporanInformasiParkirWindow", "8"))
        item = self.tableWidget.verticalHeaderItem(8)
        item.setText(_translate("LaporanInformasiParkirWindow", "9"))
        item = self.tableWidget.verticalHeaderItem(9)
        item.setText(_translate("LaporanInformasiParkirWindow", "10"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("LaporanInformasiParkirWindow", "Nomor Plat Kendaraan"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("LaporanInformasiParkirWindow", "Lokasi Tempat Parkir"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("LaporanInformasiParkirWindow", "Waktu Masuk / Keluar"))
        __sortingEnabled = self.tableWidget.isSortingEnabled()
        self.tableWidget.setSortingEnabled(False)
        self.tableWidget.setSortingEnabled(__sortingEnabled)
        
class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Create an instance of the generated UI class
        self.ui = Ui_mainWindow()
        self.ui.setupUi(self)
        
class InputCitraWindow(QMainWindow):
    def __init__(self, main_window_instance):
        super().__init__()
        # Create an instance of the generated UI class
        self.main_window = main_window_instance
        self.ui = Ui_InputCitraWindow(self,main_window_instance)
        self.ui.setupUi(self)

class InputPemilikWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Create an instance of the generated UI class
        self.ui = Ui_InputPemilikWindow()
        self.ui.setupUi(self)
        
class LaporanHasilIdentifikasiWindow(QMainWindow):
    def __init__(self,identification_data,plate_number,location,names):
        super().__init__()
        # Create an instance of the generated UI class
        self.ui = Ui_LaporanHasilIdentifikasiWindow(identification_data,plate_number,location,names)
        self.ui.setupUi(self)

class LaporanInformasiParkirWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Create an instance of the generated UI class
        self.ui = Ui_LaporanInformasiParkirWindow()
        self.ui.setupUi(self)



    
main_window_opened = False
lock_file = QLockFile("my_app_lockfile")

def check_if_already_running():
    if lock_file.tryLock(100):
        # If the lock was acquired, no other instance is running
        return False
    else:
        app_name = "SistemManajemenParkirSMKKristenImmanuel"  # Replace this with your application name
        socket_name = f"{app_name}_Socket"

        # Try connecting to the local socket server
        socket = QLocalSocket()
        socket.connectToServer(socket_name, QIODevice.WriteOnly)

        if socket.waitForConnected(500):
            # Connection successful; another instance of the application is running
            print("Application is already running.")
            return True
        else:
            # Failed to connect; no other instance is running
            server = QLocalServer()
            server.listen(socket_name)
            return False

def main():
    global main_window_opened
    
    if not main_window_opened:
        w_app_manajemen_parkir = MyMainWindow()
        #w.resize(840,400)
        #w.setWindowTitle("Sistem Manajemen Prkir SMK Kristen Immanuel")    
        print('mainnnnnnnnnnnnnnnnn')
        w_app_manajemen_parkir.show()
        main_window_opened = True
        sys.exit(app_manajemen_parkir.exec_())

if __name__ == "__main__":
    app_manajemen_parkir = QApplication([])
    app_manajemen_parkir.setApplicationName("SistemManajemenParkirSMKKristenImmanuel")
    
    if check_if_already_running():
        app_manajemen_parkir.exit(0)  # Exit if the application is already running
    else:
       main()
    
       
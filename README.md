# OBJECTIVE :
Develop a machine learning model for face emotion detection that accurately identifies and classifies facial expressions in real-time images or video frames. The goal is to create a system capable of recognizing primary emotions such as happiness, sadness, anger, surprise, fear, and disgust.

# SOFTWARE REQUIREMENTS :
### 1. Anaconda
### 2. Jupyter Notebook
### 3. Python
### 4. Python Libraries
	 Matplotlib, OpenCV-Python, Keras, Tensorflow
   
  
### 5 Google Colab
### 6 PyCharm


# CNN LAYERS EXPLANATION :

## 1.Sequential Model:
### model = Sequential()
The model is built using the Sequential API, indicating a linear stack of layers. 

## 2. Input Layer:
### Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1))
The first layer is a convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation. It expects input images of 48x48 pixels with a single channel (grayscale).'Same' padding preserves spatial dimensions. 

## 3. Convolutional Blocks:
### model.add(Conv2D(64,(3,3), padding='same', activation='relu' ))
a. Two convolutional layers with ReLU activation. 
### model.add(BatchNormalization())
b. Batch normalization to stabilize learning.
### model.add(MaxPooling2D(pool_size=(2, 2)))
c. Max pooling with pool size (2, 2) for spatial downsampling.
### model.add(Dropout(0.25))
d. Dropout with a rate of 0.25 to prevent overfitting.
### model.add(Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
e. L2 kernel regularization (0.01) in later blocks to further reduce overfitting. 


## 4. Fully Connected Layers:
### model.add(Flatten())
a. Flatten(): Flattens the 3D output from convolutional layers into a 1D vector for dense layers.
### model.add(Dense(256,activation = 'relu'))
### model.add(Dense(512,activation = 'relu'))
b. Two dense layers with 256 and 512 neurons, ReLU activation
### model.add(BatchNormalization())
### model.add(Dropout(0.25))
c. batchnormalization, and dropout (0.25).
### model.add(Dense(7, activation='softmax'))
d. Output layer with 7 neurons (for 7 emotion classes) and softmax activation for probability distribution. 
## 5. Compilation:
### model.compile(loss='categorical_crossentropy',optimizer='Adam', metrics=['accuracy']):
a. Categorical crossentropy loss appropriate for multi-class classification. Adam optimizer with a learning rate of 0.0001 and weight decay of 1e-6 for optimization.Accuracy as the evaluation metric.


# ACCURACY AND LOSS :
![ACCURACY AND LOSS](https://github.com/sumitbehera1508/Face_Emotion_Detection_CNN/assets/100491275/551d6f06-9d48-4785-85f2-7285c83419cc)

# OUTPUT :
![OUTPUT](https://github.com/sumitbehera1508/Face_Emotion_Detection_CNN/assets/100491275/69921509-11dc-44b2-adf5-9e21e6e14460)


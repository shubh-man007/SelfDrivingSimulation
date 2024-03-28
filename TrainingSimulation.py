print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from Utilis import *
from sklearn.model_selection import train_test_split

#STEP 1: Importing Data from Simulator
path = 'MyData'
data = importDataInfo(path)  #We will be using only center data to train

#STEP 2: Visualizing Data and removing redundant steering angles
data = balanceData(data , display = False)

#STEP 3: Removing columns apart from ImagePath and Steering angle
imagesPath, steerings = loadData(path,data)
print(imagesPath[0], steerings[0])

#STEP 4: Splitting data for Training and Testing
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2,random_state=5)
print('Total Training Images: ',len(xTrain))
print('Total Validation Images: ',len(xVal))

#STEP 5: To create CNN using keras
model = createModel()
model.summary()

#STEP 6: Training the Neural Network
history = model.fit(batchGen(xTrain,yTrain,100,1),steps_per_epoch=300,epochs=10,
          validation_data=batchGen(xVal,yVal,100,0),validation_steps=200)

#STEP 7: Visualizing loss and validation
model.save('model.h5')
print('Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()

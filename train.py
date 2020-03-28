import h5py
from sklearn.model_selection import train_test_split
import numpy as np
from modelLib import dummynet, get_unet, mini_unet
from utilsTrain import generator
import os
from keras.callbacks import ModelCheckpoint
from network import *

h5path = "./data/data4x.h5"
h5file = h5py.File(h5path, "r")

weightsFolder = "./weights/"
modelName = "UNET"
bestModelPath = "./weights/" + modelName + "/best.hdf5"

batchSize = 32
epochs = 100



n = h5file["image"].shape[0]
a = np.arange(n)

train, test = train_test_split(a, test_size=0.2, random_state=42)

train_generator = generator(h5file, train, batchSize)

test_generator = generator(h5file, test, batchSize)

# Define the Model

#model = dummynet()
#model = mini_unet(32,32)
model = get_HR_Att_unet(input_dim=(32, 32, 1), output_dim=(32, 32, 1), num_output_classes=1)

model.summary()

# Compile the Model & Configure

# Fit the Model

#x,y = next(train_generator)
#model.fit(x,y)

#check1 = ModelCheckpoint(os.path.join(weightsFolder, modelName + "_{epoch:02d}-loss-{val_loss:.3f}.hdf5"), monitor='val_loss', save_best_only=True, mode='auto')
check2 = ModelCheckpoint(bestModelPath, monitor='val_loss', save_best_only=True, mode='auto')
#check3 = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=int(options.patience), verbose=0, mode='auto')
#check4 = CSVLogger(os.path.join(modelFolder, modelName +'_trainingLog.csv'), separator=',', append=True)
#check5 = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=int(options.patience)//1.5, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=1e-10)



#print("\nInitiating Training:\n")
trained_model = model.fit_generator(train_generator, steps_per_epoch=(len(train) // batchSize), epochs=epochs,
                                    validation_data= test_generator, validation_steps=(len(test) // batchSize), callbacks=[check2],
                                    verbose=1)
#trained_model = model.fit_generator(train_generator, batchSize=32, epochs=epochs,
                                    #validation_data= test_generator, validation_steps=(len(test) // batchSize), callbacks=[check2],
                                    #verbose=1)


# Plot metrics 

# cleanup

train_generator.close()
test_generator.close()
h5file.close()
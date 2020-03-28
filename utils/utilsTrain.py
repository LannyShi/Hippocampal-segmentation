import numpy as np

def generator(h5file, indexes, batch_size):
    X = []
    Y = []
    idx = 0
    while True:
        for index in indexes:
            if(idx==0):
                X = []
                Y = []
            #print("Loading image : " + str(index) + " , idx = " + str(idx))
            img = np.expand_dims(h5file["image"][index], axis = 2)
            mask = np.expand_dims(h5file["mask"][index], axis = 2)
            X.append(img)
            Y.append(mask)
            idx = idx + 1
            if(idx>=batch_size):
                #print("yielding")
                idx = 0
                yield np.asarray(X),np.asarray(Y)
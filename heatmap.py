import random
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes
from matplotlib.font_manager import FontProperties
from modelLib import dummynet, get_unet, mini_unet
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.models import Model


def conv_output(model, layer_name, img):
    input_img = model.input
    try:
        out_conv = model.get_layer(layer_name).output
    except:
        raise Exception('Not layer named {}'. format(layer_name))

    intermediate_layer_model = model(inputs=input_img, outputs=out_conv)
    intermediate_output = intermediate_layer_model.predict(img)
    return intermediate_output[0]

def draw():
    xLabel = ['A', 'B', 'C', 'D', 'E']
    yLabel = ['1', '2', '3', '4', '5']

    data = []
    for i in range(5):
        temp = []
        for j in range(5):
            k = random.randint(0, 100)
            temp.append(k)
        data.append(temp)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yticks(range(len(yLabel)))
    #ax.set_yticklabels(yLabel, fontprpoperties=font)
    ax.set_xticks(range(len(xLabel)))
    ax.set_xticklabels(xLabel)
    im = ax.imshow(conv_output(mini_unet, conv4, ), cmap=plt.cm.hot_r)
    plt.colorbar(im)
    #plt.title("this is a title", fontproperties=font)
    plt.show()

d = draw()



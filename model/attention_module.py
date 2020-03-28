from keras.layers import Conv2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers.core import Layer
from keras.layers import multiply, concatenate, Reshape, Permute, Dense
import tensorflow as tf

tfe = tf.contrib.eager

class LFSAM(Layer):
    def __init__(self, **kwargs):
        super(LFSAM, self).__init__(**kwargs)
        self.gamma = tfe.Variable(0., trainable=True, name="gamma")
        self.flatten = tf.keras.layers.Flatten()

    def call(self, x, training=None, mask=None):
        # f = self.f(x)
        #print(x.shape, 'x') #（None, 32, 32, 64）
        x = Permute((3, 2, 1))(x)
        #print(x.shape, 'x_new') #（None, 64, 32, 32）
        number_of_filters = x.shape[1].value
        print('use LFSAM module')
        # print(number_of_filters)

        f = Conv2D(number_of_filters//8, (1, 1), activation='relu', padding='same', data_format='channels_first')(x)
        g = Conv2D(number_of_filters//8, (1, 1), activation='relu', padding='same', data_format='channels_first')(x)
        h = Conv2D(number_of_filters, (1, 1), activation='relu', padding='same', data_format='channels_first')(x)
        #print(f.shape, 'f') #（None, 8, 32, 32）
        #print(g.shape, 'g')  #（None, 8, 32, 32）
        #print(h.shape, 'h')  #（None, 64, 32, 32）
        f_flatten = Reshape((f.shape[1].value, f.shape[2].value*f.shape[3].value))(f)
        g_flatten = Reshape((g.shape[1].value, g.shape[2].value*g.shape[3].value))(g)
        h_flatten = Reshape((h.shape[1].value, h.shape[2].value*h.shape[3].value))(h)
        #print(f_flatten.shape, 'f_flatten')  #（None, 8, 1024）
        #print(g_flatten.shape, 'g_flatten')  #（None, 8, 1024）
        #print(h_flatten.shape, 'h_flatten')  #（None, 64, 1024）
        s = tf.matmul(g_flatten, f_flatten, transpose_a=True)  # [B,N,C] * [B, C, N] = [B, N, N]
        #print(s.shape, 's')  #（None, 1024, 1024）
        b = tf.nn.softmax(s, axis=-1)  # attention
        o = tf.matmul(h_flatten, b)
        y = self.gamma * tf.reshape(o, tf.shape(x)) + x
        #print(y.shape, 'y') #（None, 64, 32, 32）
        y = Permute((3, 2, 1))(y)
        #print(y.shape, 'y_new') #（None, 32, 32, 64）
        return y

class HFCAM(Layer):
    def __init__(self, **kwargs):
        super(HFCAM, self).__init__(**kwargs)
        self.gamma = tfe.Variable(0., trainable=True, name="gamma")
        self.flatten = tf.keras.layers.Flatten()

    def call(self, x, training=None, mask=None):
        # f = self.f(x)
        #print(x.shape, 'x') #（None, 32, 32, 64）
        x = Permute((3, 2, 1))(x)
        #print(x.shape, 'x_new') #（None, 64, 32, 32）
        number_of_filters = x.shape[1].value
        print('use HFCAM module')
        # print(number_of_filters)

        p = Reshape((x.shape[1].value, x.shape[2].value * x.shape[3].value))(x)
        #print(p.shape, 'p')  #（None, C, HW）
        q = Permute((2, 1))(p)  #（None, HW, C）
        #print(q.shape, 'q')  #（None, HW, C）
        s = tf.matmul(p, q)  # [B,C,N] * [B, N, C] = [B, C, C]
        m = tf.nn.softmax(s, axis=-1) # attention
        n = tf.matmul(q, m)
        n_new = Permute((2, 1))(n)
        #print(n_new.shape, 'n_new')
        out = self.gamma * tf.reshape(n_new, tf.shape(x)) + x
        out_new = Permute((3, 2, 1))(out)
        #print(out_new.shape, 'out_new')
        return out_new

def CCAM(inputs1, inputs2):

    #print(inputs1.shape, 'inputs1')  # （None, 32, 32, 64）
    inputs1 = Permute((3, 2, 1))(inputs1)
    #print(inputs1.shape, 'inputs1_new')  # （None, 64, 32, 32）
    #print(inputs2.shape, 'inputs2')  # （None, 32, 32, 64）
    inputs2 = Permute((3, 2, 1))(inputs2)
    #print(inputs2.shape, 'inputs2_new')  # （None, 64, 32, 32）
    x = GlobalAveragePooling2D(data_format='channels_first')(inputs1)
    y = GlobalMaxPooling2D(data_format='channels_first')(inputs1)
    #print(x.shape, 'x')
    #print(y.shape, 'y')
    print('use CCAM module')
    # x = Reshape((inputs.shape[1].value, 1, 1))(x)    # 用卷积的话用这行
    x1 = Reshape((1, 1, inputs1.shape[1].value))(x)  # 用dense的话用这行
    y1 = Reshape((1, 1, inputs1.shape[1].value))(y)  # 用dense的话用这行
    #print(x1.shape, 'x1')
    #print(y1.shape, 'y1')
    x_y = concatenate([x1, y1])
    i = inputs1.shape[1].value
    i /= 4
    i = int(i)
    # print(i)
    x2 = Dense(i, activation='relu')(x_y)
    x3 = Dense(inputs1.shape[1].value, activation='sigmoid')(x2)
    #print(x3.shape, 'x3')
    x4 = Reshape((inputs1.shape[1].value, 1, 1))(x3)
    #print(x4.shape, 'x4')
    x5 = Conv2D(inputs1.shape[1].value//2, (1, 1), activation='relu', padding='same', data_format='channels_first')(x4)
    #print(x5.shape, 'x5')
    #print(inputs2.shape, 'inputs2')
    out = multiply([inputs2, x5])
    out = Permute((3, 2, 1))(out)
    #print(out.shape, 'out')
    # output = add([out, inputs])
    return out
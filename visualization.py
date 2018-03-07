from __future__ import print_function

from scipy.misc import imsave
import numpy as np
import time
import math
from keras import backend as K
from keras.models import load_model

# dimensions of the generated pictures for each filter.
img_width = 192
img_height = 192

# the name of the layer we want to visualize
layer_name = 'conv2d_6'

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# load model
model = load_model('saved_models/inception-9733.h5')

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
print(layer_dict.keys())

for layer_name in layer_dict.keys():
    if not layer_name.startswith('conv2d'):
        continue
    if (layer_name.startswith('conv2d_22') or
        layer_name.startswith('conv2d_45') or
        layer_name.startswith('conv2d_56') or
        layer_name.startswith('conv2d_61') or
        layer_name.startswith('conv2d_83')):
        continue

    # this is the placeholder for the input images
    input_img = model.input

    def normalize(x):
        # utility function to normalize a tensor by its L2 norm
        return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

    kept_filters = []
    if K.image_data_format() == 'channels_first':
        nfilters = int(layer_dict[layer_name].output.shape[1])
    else:
        nfilters = int(layer_dict[layer_name].output.shape[3])
    n = min(math.ceil(float(nfilters ** (1.0 / 2))), 8)

    for filter_index in range(min(nfilters, n * n)):
        print('Processing filter %d' % filter_index)
        start_time = time.time()

        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        layer_output = layer_dict[layer_name].output
        if K.image_data_format() == 'channels_first':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])

        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 1.

        # we start from a gray image with some random noise
        if K.image_data_format() == 'channels_first':
            input_img_data = np.random.random((1, 3, img_width, img_height))
        else:
            input_img_data = np.random.random((1, img_width, img_height, 3))
        input_img_data = (input_img_data - 0.5) * 20 + 128

        # we run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

            print('Current loss value:', loss_value)
            if loss_value <= 0:
                # some filters get stuck to 0, we can skip them
                break

        # decode the resulting input image
        if loss_value > 0 or True:
            img = deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))
        end_time = time.time()
        print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep max the top 64 filters.
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:n * n]

    # build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    margin = 5
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            if idx < nfilters:
                img, loss = kept_filters[i * n + j]
                stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                                 (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    # save the result to disk
    imsave('filters/%s_filters_%dx%d.png' % (layer_name, n, n), stitched_filters)

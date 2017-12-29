# Simple program to recognize specific shape among other shapes on the image.
# Images are automatically generated so you don't need to look for a dataset.
# Every image contains single rectangle and several triangle, the program tries to predict location of a rectangle.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


NUM_IMAGES = 50000

IMAGE_SIZE = 24
MIN_SHAPE_SIZE = 3
MAX_SHAPE_SIZE = 6

NUM_OTHER_SHAPES = 1

rectangle_boxes = np.zeros((NUM_IMAGES, 1, 4))
images = np.zeros((NUM_IMAGES, IMAGE_SIZE, IMAGE_SIZE))


def create_shape_to_recognize(i):
    # create rectangle
    width, height = np.random.randint(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE, size=2)
    x = np.random.randint(0, IMAGE_SIZE - width)
    y = np.random.randint(0, IMAGE_SIZE - height)
    images[i, x:x + width, y:y + height] = 1.
    rectangle_boxes[i, 0] = [x, y, width, height]


def create_other_shape(i):
    # create triangle
    size = np.random.randint(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)
    x, y = np.random.randint(0, IMAGE_SIZE - size, size=2)
    mask = np.tril_indices(size)
    images[i, x + mask[0], y + mask[1]] = 1.


for image_id in range(NUM_IMAGES):
    create_shape_to_recognize(image_id)

    for _ in range(NUM_OTHER_SHAPES):
        create_other_shape(image_id)


def display_image(image_id):
    plt.imshow(images[image_id].T, cmap='Greys', interpolation='none', origin='lower', extent=[0, IMAGE_SIZE, 0, IMAGE_SIZE])
    rectangle_shape = rectangle_boxes[image_id, 0]
    plt.gca().add_patch(
        matplotlib.patches.Rectangle((rectangle_shape[0], rectangle_shape[1]), rectangle_shape[2], rectangle_shape[3],
                                     ec='r', fc='none'))
    # displays image and blocks
    plt.show()


display_image(0)


# Reshape and normalize the image data to mean 0 and std 1.
X = (images.reshape(NUM_IMAGES, -1) - np.mean(images)) / np.std(images)
X.shape, np.mean(X), np.std(X)


# Normalize x, y, w, h by img_size, so that all values are between 0 and 1.
# Important: Do not shift to negative values (e.g. by setting to mean 0), because the IOU calculation needs positive w and h.
y = rectangle_boxes.reshape(NUM_IMAGES, -1) / IMAGE_SIZE
y.shape, np.mean(y), np.std(y)

# Split training and test.
i = int(0.8 * NUM_IMAGES)
train_X = X[:i]
test_X = X[i:]
train_y = y[:i]
test_y = y[i:]
test_imgs = images[i:]
test_bboxes = rectangle_boxes[i:]

# Build the model.
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

model = Sequential([
        Dense(200, input_dim=X.shape[-1]),
        Activation('relu'),
        Dropout(0.2),
        Dense(y.shape[-1])
    ])
model.compile('adadelta', 'mse')


# Train.
model.fit(train_X, train_y, nb_epoch=30, validation_data=(test_X, test_y), verbose=2)

# Predict bounding boxes on the test images.
pred_y = model.predict(test_X)
pred_bboxes = pred_y * IMAGE_SIZE
pred_bboxes = pred_bboxes.reshape(len(pred_bboxes), 1, -1)


# Intersection Over Union and measures the overlap between the predicted and the real bounding box.
# It’s calculated by dividing the area of intersection (red in the image below) by the area of union (blue).
# The IOU is between 0 (no overlap) and 1 (perfect overlap).
# Calculate overlap between two bounding boxes [x, y, w, h] as the area of intersection over the area of unity
def IOU(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

    w_I = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_I = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_I <= 0 or h_I <= 0:  # no overlap
        return 0.
    I = w_I * h_I

    U = w1 * h1 + w2 * h2 - I

    return I / U


# Show a few images and predicted bounding boxes from the test dataset.
plt.figure(figsize=(12, 3))
for i_subplot in range(1, 5):
    plt.subplot(1, 4, i_subplot)
    i = np.random.randint(len(test_imgs))
    plt.imshow(test_imgs[i].T, cmap='Greys', interpolation='none', origin='lower', extent=[0, IMAGE_SIZE, 0, IMAGE_SIZE])
    for pred_bbox, exp_bbox in zip(pred_bboxes[i], test_bboxes[i]):
        plt.gca().add_patch(
            matplotlib.patches.Rectangle((pred_bbox[0], pred_bbox[1]), pred_bbox[2], pred_bbox[3], ec='r', fc='none'))
        plt.annotate('IOU: {:.2f}'.format(IOU(pred_bbox, exp_bbox)), (pred_bbox[0], pred_bbox[1] + pred_bbox[3] + 0.2),
                     color='r')
    plt.show()
    # plt.savefig('plots/bw-single-rectangle_prediction.png', dpi=300)

# Calculate the mean IOU (overlap) between the predicted and expected bounding boxes on the test dataset.
summed_IOU = 0.
for pred_bbox, test_bbox in zip(pred_bboxes.reshape(-1, 4), test_bboxes.reshape(-1, 4)):
    summed_IOU += IOU(pred_bbox, test_bbox)
mean_IOU = summed_IOU / len(pred_bboxes)

print("Mean IOU: ", mean_IOU)
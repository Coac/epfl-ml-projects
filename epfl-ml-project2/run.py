import os

import keras
import scipy
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Input
from keras.models import Model
from keras.models import Sequential

from functions_v1 import *
from mask_to_submission import *

if __name__ == '__main__':

    padding = 24
    patch_size = 16

    input_shape = (64, 64, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(128))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    model.load_weights("./weights/model.h5")

    autoencoder_padding = 13


    def get_autoencoder_XY(model, X, Y_labels, image_size, patch_size, autoencoder_padding, data_augmentation=False):
        pred_img_array = []
        gt_image_array = []
        patch_count_per_image = int((image_size / patch_size) ** 2)
        for i in range(int(len(X) / patch_count_per_image)):
            pred_img, gt_image = display_prediction_and_gt(model, X, Y_labels, image_size, patch_size, i, display=False)

            pred_img = pred_img.reshape(image_size, image_size, 1)
            pred_img = pad_image(pred_img, autoencoder_padding)

            if Y_labels is not None:
                gt_image = gt_image.reshape(image_size, image_size, 1)
                gt_image = pad_image(gt_image, autoencoder_padding)

            if data_augmentation:
                for j in range(4):
                    pred_img_array.append(np.rot90(np.flip(pred_img, 0), j))
                    pred_img_array.append(np.rot90(pred_img, j + 1))

                if Y_labels is not None:
                    for j in range(4):
                        gt_image_array.append(np.rot90(np.flip(gt_image, 0), j))
                        gt_image_array.append(np.rot90(gt_image, j + 1))
            else:
                pred_img_array.append(pred_img)

                if Y_labels is not None:
                    gt_image_array.append(gt_image)

        pred_img_array = np.array(pred_img_array)

        if Y_labels is not None:
            gt_image_array = np.array(gt_image_array)
        else:
            gt_image_array = None

        return pred_img_array, gt_image_array


    # Autoencoder
    input_img = Input(shape=(76, 76, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.2)(x)

    # at this point the representation is (7, 7, 32)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.load_weights("./weights/autoencoder.h5")

    TEST_IMG_SIZE = 608

    test_dir = "datas/test_set_images/"
    files = os.listdir(test_dir)
    n = min(100, len(files))  # Load maximum 20 images
    files.sort()
    print("Loading " + str(n) + " images")
    test_imgs = [load_image(test_dir + files[i] + '/' + os.listdir(test_dir + files[i])[0]) for i in range(n)]
    print("Loaded !")

    test_padded_images = pad_list_image(test_imgs, padding)
    test_patches = [get_patches(image, patch_size, patch_size, padding) for image in test_padded_images]
    test_patches = [patch for image_patches in test_patches for patch in
                    image_patches]  # Flatten the array of patches to array of patch

    test_X = np.array(test_patches)

    test_X_autoencoder, _ = get_autoencoder_XY(model, test_X, None, 76, 2, 0)

    index = 0

    # Enlarge image and save it
    for i in range(50):
        decoded_img = autoencoder.predict(np.array([test_X_autoencoder[i]]))[0]
        enlarged_image = scipy.misc.imresize(decoded_img.reshape((76, 76)), size=(608, 608))
        prediction_test_dir = "predictions/"

        im = Image.fromarray(img_float_to_uint8(enlarged_image))
        print("Saving " + prediction_test_dir + "prediction_" + str(i + 1) + ".png")

        im.save(prediction_test_dir + "prediction_" + str(i + 1) + ".png")

    # Generate submission
    submission_filename = 'submission.csv'
    image_filenames = []
    for i in range(1, 51):
        image_filename = 'predictions/prediction_' + str(i) + '.png'
        image_filenames.append(image_filename)
    masks_to_submission(submission_filename, *image_filenames)

    print('submission.csv created !')

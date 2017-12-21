import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_image(infilename):
    """ Read an image from a directory
    INPUTS: 
        infilename: Name of the image (include directory)
    OUTPUT: 
        data = image file (float 32)
    """
    data = mpimg.imread(infilename)
    return data


def load_image_pil(infilename, bw=False):
    """ Read an image from a directory
    INPUTS: 
        infilename: Name of the image (include directory)
        bw = Boolean (1 color image, 0 black and white)
    OUTPUT: 
        data = image file (float 32)
    """
    img = Image.open(infilename)
    if bw:
        img = img.convert('L')

    img = np.array(img)

    return img


def img_float_to_uint8(img):
    """Change image from float to uint8 0-255
    INPUTS: 
        img = image in float format
    OUTPUTS:
        rimg = image in uint8 format
    """
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


def concatenate_images(img, gt_img):
    """Concatenate an image and its groundtruth
    INPUTS: 
        img = RGB image
        gt_img = Groundtruth of image
    OUTPUTS:
        cimg = concatenated image (w,h,dim)
    """
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def get_patches(im, w, h, padding):
    """Get patches of size w x h of images
    INPUTS:
        im = image 
        w = width of patch
        h = height of patch
        padding = extra outside of image to give context
    OUTPUTS: 
        list_patches = List of all patches of images with their padding 
                        ((w+2*padding) x (h+2*padding))
    """
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight - padding * 2, h):
        for j in range(0, imgwidth - padding * 2, w):
            if is_2d:
                im_patch = im[j:j + w + padding * 2, i:i + h + padding * 2]
            else:
                im_patch = im[j:j + w + padding * 2, i:i + h + padding * 2, :]
            list_patches.append(im_patch)
    return list_patches


def patch_to_label(patch):
    """Transforms each groundtruth patch to an integer (1=road or 0=none) 
       calculating the average value of the pixels in the patch
    INPUTS: 
        patch = image patch
    OUTPUTS: 
        Returns a label (1=road or 0 = no road)
    """
    foreground_threshold = 0.25
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def label_to_img(imgwidth, imgheight, w, h, labels):
    """Gives labels to the test images
    INPUTS:
        imgwidth = with of image
        imgheight = height of image
        w = width of patch of image
        h = height of patch of image
        labels = labels returned by the ML learning model when applied to test
    OUTPUTS: 
        im = groundtruth image for the test image
    """
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            im[j:j + w, i:i + h] = labels[idx]
            idx = idx + 1
    return im


def pad_image(image, padding):
    """Give an external padding to the original image by mirroring the outside
    INPUTS: 
        image = original image of size (w x h)
        padding = padding size (p)
    OUTPUTS: 
        padded_image = image with the added reflected padding of size ((w+p) x (h+p))
    
    """
    padded_image = np.lib.pad(image, ((padding, padding), (padding, padding), (0, 0)), 'reflect')
    return padded_image


def pad_list_image(images, padding):
    """Pad a list of images
    INPUTS: 
        images = list of images
        padding = padding size
    OUTPUTS: 
        padded_images = list of padded images
    """
    padded_images = []
    for i in range(len(images)):
        image = pad_image(images[i], padding)
        padded_images.append(image)

    return padded_images


def save_labels_to_images(imgwidth, imgheight, w, h, labels):
    """Takes the predicted labels from a set and turns it into an image (uint8)
    INPUTS: 
        imgwidth = image width 
        imgheight = image height 
        w = patch width 
        h = patch height
        labels = labels of each patch
    OUTPUTS: 
        saved image in the "predictions" directory    
    """
    patch_per_image = int(imgwidth / w) * int(imgheight / h)
    count = int(len(labels) / patch_per_image)

    for i in range(count):
        img = label_to_img(imgwidth, imgheight, w, h, labels[i * patch_per_image:(i + 1) * patch_per_image])
        im = Image.fromarray(img_float_to_uint8(img))
        prediction_test_dir = "predictions/"
        print("Saving " + prediction_test_dir + "prediction_" + str(i + 1) + ".png")
        im.save(prediction_test_dir + "prediction_" + str(i + 1) + ".png")


def display_prediction_and_gt(model, X, Y, image_size, patch_size, image_index, display=True):
    """Displays the image of the prediction and the test groundtruth next to each
        other to compare visually
    INPUTS: 
        model = Model weights
        X = List of patches of images
        Y = List of labels of images
        image_size = Image size
        patch_size = Patch size
        imge_index = Image index to display
    OUTPUTS: 
        plot of predicted and real grountruth image
    
    """
    patch_count_per_image = int((image_size / patch_size) ** 2)

    predictions = model.predict_classes(
        X[image_index * patch_count_per_image: (image_index + 1) * patch_count_per_image])

    pred_img = label_to_img(image_size, image_size, patch_size, patch_size, predictions)

    if Y is not None:
        gt_image = label_to_img(image_size, image_size, patch_size, patch_size,
                                Y[image_index * patch_count_per_image: (image_index + 1) * patch_count_per_image])
    else:
        gt_image = None

    if display:
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(pred_img, cmap='Greys_r');
        if Y is not None:
            plt.subplot(1, 2, 2)
            plt.imshow(gt_image, cmap='Greys_r');

    return pred_img, gt_image


def categorical_probas_to_classes(p):
    return np.argmax(p, axis=1)


def probas_to_classes(y_pred):
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        return categorical_probas_to_classes(y_pred)
    return np.array([1 if p > 0.5 else 0 for p in y_pred])


def show_ZCA(im, h, w, bw=False):
    if bw:
        im = im.reshape((h, w, 1))
    else:
        im = im.reshape((h, w, 3))
    m, M = im.min(), im.max()
    plt.imshow((im - m) / (M - m))
    plt.show()


def zca_whitening(images, h, w, n_batch, bw=False):
    """
    Function to compute ZCA whitening matrix .
    INPUT:  
        images: [M x N] matrix.
            Rows: Variables
            Columns: Observations
        h: height of images
        w: width of image
        n_batch: Number of batches
        bw: black and white image
    OUTPUT: 
        ZCAMatrix: [M x M] matrix
        flat_zca: vector of whitened images
    """
    # Get a batch size
    batch = int(len(images) / n_batch)

    # Convert the list into a numpy array
    images = np.array(images)

    # Flatten image
    if bw:
        images = images.reshape((-1, h * w))
    else:
        images = images.reshape((-1, h * w * 3))

    print(images.shape)

    # 0 center the image
    images = images - images.mean(axis=0)
    # Global Contrast Normalization, which is quite often applied to image data. I'll use the L2 norm, which makes every image have vector magnitude 1:
    images = images / np.sqrt((images ** 2).sum(axis=1))[:, None]

    ZCA_matrix = []
    for i in range(n_batch):
        print("ZCA batch {i} out of {n_batch}".format(i=i + 1, n_batch=n_batch))
        # Get a batch of the images
        im_batch = images[batch * i:(i + 1) * batch]

        # Covariance matrix
        sigma = np.cov(im_batch, rowvar=True)
        #         print(sigma.shape)
        # Singular Value Decomposition
        U, S, V = np.linalg.svd(sigma)
        # Whitening constant, it prevents division by zero
        epsilon = 1e-5
        # ZCA Whitening matrix
        ZCAMatrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))
        #         print(ZCAMatrix.shape)

        # Create a list of the ZCA Matrices
        ZCA_matrix.append(ZCAMatrix)

    # Calculate an average of all the ZCA matrices
    ZCAMatrix = np.mean(np.array(ZCA_matrix), axis=0)
    print("\nZCA Matrix shape = {shape}".format(shape=ZCAMatrix.shape))

    zca_ims = []
    for i in range(n_batch):
        # Get a batch of the images
        im_batch = images[batch * i:(i + 1) * batch]

        # Data whitening
        zca = np.dot(ZCAMatrix, im_batch)

        # Turn into values between 0 and 1
        m, M = zca.min(axis=0), zca.max(axis=0)
        zca = (zca - m) / (M - m)

        # Return the image to its shape
        if bw:
            zca = zca.reshape((-1, h, w, 1))
        else:
            zca = zca.reshape((-1, h, w, 3))

        zca_ims.append(zca)

    # Unlist the small lists to make a unique list of the whitened images
    flat_zca = [item for sublist in zca_ims for item in sublist]

    return ZCAMatrix, flat_zca


def zca_whitening_test(images, h, w, n_batch, ZCA_mat, bw=False):
    """
    Function to compute ZCA whitening matrix .
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix
    """
    # Get a batch size
    batch = int(len(images) / n_batch)

    # Convert the list into a numpy array
    images = np.array(images)

    # Flatten image
    if bw:
        images = images.reshape((-1, h * w))
    else:
        images = images.reshape((-1, h * w * 3))

    print(images.shape)

    # 0 center the image
    images = images - images.mean(axis=0)
    # Global Contrast Normalization, which is quite often applied to image data. I'll use the L2 norm, which makes
    # every image have vector magnitude 1:
    images = images / np.sqrt((images ** 2).sum(axis=1))[:, None]

    zca_ims = []
    for i in range(n_batch):
        if i == n_batch - 1:
            # Get a batch of the images
            im_batch = images[batch * i:]

            # Data whitening
            zca = np.dot(ZCAMatrix[0:im_batch.shape[0], 0:im_batch.shape[0]], im_batch)
        else:
            # Get a batch of the images
            im_batch = images[batch * i:(i + 1) * batch]

            # Data whitening
            zca = np.dot(ZCAMatrix, im_batch)

            # Turn into values between 0 and 1
        m, M = zca.min(axis=0), zca.max(axis=0)
        zca = (zca - m) / (M - m)

        # Return the image to its shape
        if bw:
            zca = zca.reshape((-1, h, w, 1))
        else:
            zca = zca.reshape((-1, h, w, 3))

        zca_ims.append(zca)

    # Unlist the small lists to make a unique list of the whitened images
    flat_zca = [item for sublist in zca_ims for item in sublist]

    return np.array(flat_zca)

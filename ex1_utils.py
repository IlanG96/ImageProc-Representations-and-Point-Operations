"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import matplotlib.pyplot as plt
from typing import List
import cv2
import numpy as np
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 1111111


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    image = cv2.imread(filename)
    if representation == LOAD_GRAY_SCALE:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif representation == LOAD_RGB:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    image = image.astype(np.float32) / 255.0
    return image
    


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    image = imReadAndConvert(filename, representation)
    plt.imshow(image)
    # If the given representation is gray, make the image gray
    if representation is LOAD_GRAY_SCALE:
        plt.gray()
    
    plt.show()
    


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    conversion_matrix = np.array( [[0.299, 0.587, 0.114],
                                  [0.596, -0.275, -0.321],
                                  [0.212, -0.523, 0.311]])

    return imgRGB @ conversion_matrix.T                          



def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    
    conversion_matrix = np.array( [[0.299, 0.587, 0.114],
                                  [0.596, -0.275, -0.321],
                                  [0.212, -0.523, 0.311]])
    conversion_matrix = np.linalg.inv(conversion_matrix)                              
    
    return imgYIQ @ conversion_matrix.T


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Equalizes the histogram of an image
    :param imgOrig: Original image in RGB or grayscale
    :return: Tuple containing the equalized image, the original histogram, and the equalized histogram
    """
    # Convert image to YIQ color space if it's RGB
    if len(imgOrig.shape) > 2:
        YIQImg = transformRGB2YIQ(imgOrig)
        ychannel = YIQImg[:,:,0] # Extract the Y channel
    else:
        ychannel = imgOrig

    # Make a copy of the Y channel for later use
    origYChannel = ychannel.copy()

    # Normalize Y channel to 0-255 range
    ychannel = cv2.normalize(ychannel, None, 0, 255, cv2.NORM_MINMAX)

    # Flatten the Y channel for easier histogram calculation
    ychannel = ychannel.astype(np.int).flatten()

    # Calculate the histogram of the Y channel
    histOrg, bins = np.histogram(ychannel, 256, (0, 255))

    # Calculate the cumulative sum of the histogram
    cumsum = np.cumsum(histOrg)

    # Calculate the lookup table (LUT) for mapping pixel values to new values
    LUT = [np.ceil((i*np.max(origYChannel) / cumsum[-1]) * 255) for i in cumsum]

    # Map the pixel values in the Y channel using the LUT
    newImg = np.array([LUT[pixel] for pixel in ychannel])

    # Reshape the new Y channel to the original shape
    newImg = newImg.reshape(origYChannel.shape)

    # Calculate the histogram of the equalized Y channel
    histEQ, bins = np.histogram(newImg, 256, (0, 255))

    # Normalize the new Y channel to 0-1 range
    newImg = newImg/255

    # Convert back to RGB color space if it was RGB originally
    if len(imgOrig.shape) > 2:
        YIQImg[:,:,0] = newImg # Replace the old Y channel with the equalized Y channel
        newImg = transformYIQ2RGB(YIQImg)

    # Return the equalized image, the original histogram, and the equalized histogram
    return newImg, histOrg, histEQ
  
    


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """

    # if the image is single-channel (grey channel), quantize it directly
    if len(imOrig.shape) == 2:
        return quantize(imOrig.copy(), nQuant, nIter)

    # convert the image from RGB to YIQ color space
    # the quantization procedure should only operate on the Y channel
    qImage_i = []
    YIQImg = transformRGB2YIQ(imOrig)
    y_channel, error_i = quantize(YIQImg[:, :, 0].copy(), nQuant, nIter)
    for img in y_channel:
        # convert the quantized image back to RGB color space
        q_image_i = transformYIQ2RGB(np.dstack((img, YIQImg[:, :, 1], YIQImg[:, :, 2])))
        qImage_i.append(q_image_i)

    return qImage_i, error_i



def quantize(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
    Quantize a single-channel image (gray channel) into **nQuant** colors
    :param imOrig: The original image (RGB or grayscale)
    :param nQuant: Number of colors to quantize the image to
    :param nIter: Number of optimization loops
    :return: (List[qImage_i],List[error_i])
    """
    # Create an array to store the quantized images and the errors
    quantized_images = []
    errors = []

    # Normalize the image to range [0, 255]
    imOrig = cv2.normalize(imOrig, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    im_flat = imOrig.ravel().astype(int)

    # Compute the histogram of the original image
    hist, edges = np.histogram(im_flat, bins=256)

    # Define the boarders for the quantized colors
    boarders = np.linspace(0, 255, num=nQuant+1).astype(int)

    # Perform the optimization loop
    for i in range(nIter):

        # Compute the weighted average intensity for each color segment
        means = []
        for j in range(nQuant):
            segment_hist = hist[boarders[j]:boarders[j+1]]
            idx = np.arange(len(segment_hist))
            mean_intensity = (segment_hist * idx).sum() / np.sum(segment_hist)
            means.append(boarders[j] + mean_intensity)

        # Create a new image with the quantized colors
        quantized_image = np.zeros_like(imOrig)
        for k in range(len(means)):
            quantized_image[imOrig > boarders[k]] = means[k]

        # Compute the error between the original and quantized images
        mse = np.sqrt((imOrig - quantized_image) ** 2).mean()
        errors.append(mse)

        # Store the quantized image
        quantized_images.append(quantized_image / 255.0)

        # Update the boarders for the next iteration
        for k in range(len(means) - 1):
            boarders[k+1] = int((means[k] + means[k+1]) / 2)

    # Return the quantized images and errors
    return quantized_images, errors





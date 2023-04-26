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
from ex1_utils import LOAD_GRAY_SCALE , LOAD_RGB
import cv2

def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    image = cv2.imread(img_path)

    # Convert the representation to grayscale or RGB
    if rep == LOAD_GRAY_SCALE:
        image  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   
    #function for updating the image based on the trackbar value
    def update_gamma(value):
        gamma = (0.02 + value / 100.0)
        gamma_corrected_image = pow(image/255, gamma)
        cv2.imshow('Gamma Correction', gamma_corrected_image)

    # Create the trackbar window and trackbar
    cv2.namedWindow('Gamma Correction')
    cv2.createTrackbar('Gamma', 'Gamma Correction', 100, 200, update_gamma)
    # Display the initial image
    cv2.imshow('Gamma Correction', image)
    cv2.waitKey(0)


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()

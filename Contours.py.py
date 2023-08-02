import argparse
import os
from argparse import Namespace

import cv2
import numpy as np

"""Sobel, Scharr, Prewitt, Canny, Laplacian - algorithms I've chosen"""
kernel = np.ones((5, 5), np.uint8)


def prepare_image(image):
    """Preparing image for further use
    :param image: original image
    :returns: Prepared image
    """

    grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(grayscale_img, (5, 5), cv2.BORDER_DEFAULT)
    return blurred_img


def sobel(image):
    """Detecting edges using sobel algorithm
    :param image: prepared image
    :returns: Image with detected edges
    """
    edges_sobel_X = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    edges_sobel_Y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    abs_edges_sobel_X = cv2.convertScaleAbs(edges_sobel_X)
    abs_edges_sobel_Y = cv2.convertScaleAbs(edges_sobel_Y)

    edges_sobel = cv2.addWeighted(abs_edges_sobel_X, 0.5, abs_edges_sobel_Y, 0.5, 0)
    return edges_sobel


def scharr(image):
    """Detecting edges using scharr algorithm
    :param image: prepared image
    :returns: Image with detected edges
    """
    scharr_X = cv2.Scharr(image, cv2.CV_64F, 1, 0, 9)
    scharr_X_abs = np.uint8(np.absolute(scharr_X))
    scharr_Y = cv2.Scharr(image, cv2.CV_64F, 0, 1, 9)
    scharr_Y_abs = np.uint8(np.absolute(scharr_Y))
    scharr_XY_combined = cv2.bitwise_or(scharr_Y_abs, scharr_X_abs)
    return scharr_XY_combined


def canny(image):
    """Detecting edges using canny algorithm
    :param image: prepared image
    :returns: Image with detected edges
    """
    edges_canny = cv2.Canny(image, 30, 90)
    return edges_canny


def laplacian(image):
    """Detecting edges using laplacian algorithm
    :param image: prepared image
    :returns: Image with detected edges
    """
    edges_laplacian = cv2.Laplacian(image, cv2.CV_64F, 4)
    image = cv2.GaussianBlur(edges_laplacian, (11, 11), 0)
    edges_laplacian = cv2.Laplacian(image, cv2.CV_64F, 4)
    image = cv2.GaussianBlur(edges_laplacian, (3, 3), 0)
    edges_laplacian = cv2.Laplacian(image, cv2.CV_64F, 4)
    return edges_laplacian


def prewitt(image):
    """Detecting edges using prewitt algorithm
    :param image: prepared image
    :returns: Image with detected edges
    """
    kernel_X = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernel_Y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    edges_prewitt_X = cv2.filter2D(image, -1, kernel_X)
    edges_prewitt_Y = cv2.filter2D(image, -1, kernel_Y)
    edges_prewitt_XY = edges_prewitt_X + edges_prewitt_Y
    return edges_prewitt_XY


def main(args: Namespace) -> None:
    """
    This is where everything is done using functions

    :param args: Namespace storing all arguments from command line
    :returns: None
    """
    path = args.directory_path
    if path is None:
        print('Wrong directory...')
        return
    for name in os.listdir(path):
        image = cv2.imread(os.path.join(path, name))
        if image is None:
            print('Error opening image')
            return
        if '_copy_' in name:
            continue
        new_image = prepare_image(image)
        if args.algorithm == 'canny':
            new_image = canny(new_image)
        elif args.algorithm == 'sobel':
            new_image = sobel(new_image)
        elif args.algorithm == 'laplacian':
            new_image = laplacian(new_image)
        elif args.algorithm == 'prewitt':
            new_image = prewitt(new_image)
        elif args.algorithm == 'scharr':
            new_image = scharr(new_image)
        else:
            args.algorithm = 'canny'
            new_image = canny(new_image)
        cv2.imshow(args.algorithm, new_image)
        cv2.waitKey(0)
        final_name = args.algorithm + '_copy_' + name
        cv2.imwrite(os.path.join(args.directory_path, final_name), new_image)

    print('Images saved as copy using: ' + args.algorithm)


def parse_arguments() -> Namespace:
    """
    This is a function that parses arguments from command line.

    :param: None
    :returns: Namespace storing all arguments from command line
    """
    parser = argparse.ArgumentParser(
        description='Detecting edges on the image - project WMA2 by s21099')
    parser.add_argument('-a',
                        '--algorithm',
                        type=str,
                        default='canny',
                        help='Input name of chosen algorithm: sobel, scharr, canny, laplacian or prewitt')
    parser.add_argument('-d',
                        '--directory_path',
                        type=str,
                        required=True,
                        help='Input a path where images are')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())

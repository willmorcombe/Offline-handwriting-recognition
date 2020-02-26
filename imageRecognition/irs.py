from PIL import Image
import numpy as np
import os, sys
import math
from scipy import ndimage

photoFileName = "image"
photoFolderName = "imageData"

def toMatrix(im, l):
    width, height = im.size
    mat = np.array(l).reshape(width, height)
    return mat

def toGreyScale(im, pixels):
    newPixels = []
    counter = 0
    width, height = im.size

    for i in range(height):
        subList = []
        for j in range(width):
            pixelVal = pixels[counter]
            if pixelVal > 110:
                pixelVal = 0
            else:
                pixelVal = 255 - pixelVal + 10
                if pixelVal > 255:
                    pixelVal = 255 - pixelVal
            subList.append(pixelVal)
            counter += 1
        newPixels.append(subList)
    return newPixels


def croppedImage(im, startPos):
    width, height = im.size
    heightImage = []
    cropImage = []
    flag = False

    im = im.crop((0, 0, width, height))

    pixels = toGreyScale(im, list(im.getdata()))

    for x in range(len(pixels)):
        if sum(pixels[x]) != 0:
            heightImage.append(pixels[x])

    pixels = np.asarray(heightImage)
    pixels = pixels.T

    for x in range(len(pixels)):
        if sum(pixels[x]) != 0:
            cropImage.append(pixels[x])
            flag = True
        elif sum(pixels[x]) == 0 and flag == True:
            cord = x
            break

    cropImage = np.asarray(cropImage)
    cropImage = np.rot90(cropImage, 0).T

    im = Image.fromarray(cropImage)

    return (im, x)


def getCenterOfMassShift(im):
    cy,cx = ndimage.measurements.center_of_mass(im)

    width, height = im.shape
    shiftx = np.round(height/2.0-cx).astype(int)
    shifty = np.round(width/2.0-cy).astype(int)

    return shiftx, shifty

def shift(im, shiftTuple):
    x, y = shiftTuple

    array = (1, 0, y, 0, 1, x)
    im = im.transform(im.size, Image.AFFINE, array)
    return im



def handWrittenNumberData():

    im = Image.open('imageRecognition' + '/' + photoFolderName + "/" + photoFileName + ".jpg").convert('L')
    pixels = list(im.getdata())
    #turns the list of pixelsto a matrix of pixels
    # pixels = toMatrix(im, list(im.getdata()))

    pixels = toGreyScale(im, pixels)
    array = np.array(pixels, dtype=np.uint8)
    formatted = Image.fromarray(array)
    #image is now black and white.

    pixelStartPos = 0
    images = []

    for x in range(3):
        im, pixelStartPos = croppedImage(formatted, pixelStartPos)
        im.show()
        images.append(im)
        print(x)

    images[0].show()
    images[1].show()
    images[2].show()
    #image is cropped to remove black rows and columns

    rows, cols = im.size

    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        im = im.resize((rows, cols))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        im = im.resize((rows, cols))

    # image fits in a 20x20 box
    # image = centreOfMass(im)
    colsPadding = (int(math.ceil((28-cols)/2.0)), int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)), int(math.floor((28-rows)/2.0)))
    pixelArray = np.lib.pad(im, (colsPadding, rowsPadding), 'constant')

    im = Image.fromarray(pixelArray)


    shiftTuple = getCenterOfMassShift(pixelArray)
    shiftedImage = shift(im, shiftTuple)
    im = shiftedImage
    im.show()

    imagePixels = list(im.getdata())

    im.save('imageRecognition' + '/' + photoFolderName + "/" + 'formattedImage.png')

    return imagePixels














# def centreOfMass(im):
#     image_pixels = list(im.getdata())
#     width, height = im.size
#     image_array = np.asarray(image_pixels).reshape(width, height)

#     sum_of_colls = [sum(x) for x in zip(*image_array)]
#     sum_of_rows = list(np.sum(image_array, axis=1))
#     sum_of_colls_index = []
#     sum_of_rows_index = []

#     zipList = zip(sum_of_colls, sum_of_rows)

#     for index, num in enumerate(zipList):
#         c, r = (index * num[0], index * num[1])
#         sum_of_colls_index.append(c)
#         sum_of_rows_index.append(r)

#     sum_of_colls = sum(sum_of_colls)
#     sum_of_rows = sum(sum_of_rows)
#     sum_of_colls_index = sum(sum_of_colls_index)
#     sum_of_rows_index = sum(sum_of_rows_index)

#     center_of_mass_y = sum_of_colls_index // sum_of_colls
#     center_of_mass_x = sum_of_rows_index // sum_of_rows


#     background = Image.new('L', ((28,28)))
#     offset_x = 8
#     offset_y = 10 - center_of_mass_y -2


#     background.paste(im, ((offset_x, offset_y)))

#     image = background

#     return image
#   # return image object

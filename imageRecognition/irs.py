from PIL import Image
import numpy as np
import os, sys
import math
from scipy import ndimage

photoFileName = "image"
photoFolderName = "imageData"

# turns the list of pixels in to an numpy matrix dependant on the size of the image

#returns the matrix of pixels
def toMatrix(im, l):
    width, height = im.size
    mat = np.array(l).reshape(height, width)
    return mat

# changes the values of the pixels depending on their brightness.
# the writing will be black and the background will be white

#returns the list of changed pixles
def toGreyScale(pixels):
    for x in range(len(pixels)):
        if pixels[x] > 100:
            pixels[x] = 0
        else:
            pixels[x] = 255

    return pixels

def getLine(im, startPos):
    width, height = im.size
    lineImage = []
    cropImage = []
    endPos = 0
    flag = False # flag turns true when the script has scanned over a single image.

    im = im.crop((0, startPos, width, height))
    im.show()
    pixels = toMatrix(im, list(im.getdata()))

    if sum(sum(pixels)) < 100_000:
        return (0, 0)

    for x in range(len(pixels)):
        if sum(pixels[x]) != 0:
            lineImage.append(pixels[x])
            flag = True
        elif sum(pixels[x]) == 0 and flag == True:
            endPos = x
            break


    line = np.asarray(lineImage)
    if line.sum() < 100_000:
        return (1, endPos)
    else:
        im = Image.fromarray(line)
        return (im, endPos)


# splits the image in to its individual characters. (Still has a black border around in some cases)
#input => the original black and white image,
#         the x starting position to make sure the photo is correctly cropped

# returns an image of a single character and an integer for cropping.
def splitImage(im, startPos):
    width, height = im.size
    heightImage = []
    cropImage = []
    flag = False # flag turns true when the script has scanned over a single image.

    im = im.crop((startPos, 0, width, height)) # crops image depending on startPos

    pixels = toMatrix(im, list(im.getdata()))
    p = np.asarray(pixels)
    im = Image.fromarray(p)
    # im.show()
    if p.sum() < 100_000:
        return (0, 0)

# looking at the sum of each row of the image. If sum is == 0 then there is no image
# so can ignore that part of the image. Its only interested when there is writing, and when
# there is, it will store the row in a new list. (heightImage)
    for x in range(len(pixels)):
        if sum(pixels[x]) != 0:
            heightImage.append(pixels[x])

# flip the image so we can use the same algorithm for looking at each row of the image.
    pixels = np.asarray(heightImage)
    pixels = pixels.T

    for x in range(len(pixels)):
        if sum(pixels[x]) != 0:
            cropImage.append(pixels[x])
            flag = True
        elif sum(pixels[x]) == 0 and flag == True:
            cord = x #the x position of the image to use for later cropping
            break

    cropImage = np.asarray(cropImage)
# conditional for checking if there isn't a character in the image.
#(there may be a black dot on the screen hence the sum of 100000)\
    if cropImage.sum() < 100_000:
        return (1, cord)
    else:
        cropImage = np.rot90(cropImage, 0).T
        im = Image.fromarray(cropImage)
        return (im, cord)

# trims the cropped image to fit in a 20 x 20 box.
# returns the cropped image

# uses same algorithm of using the sums of each column to see if there is a pixel there.
#returns the cropped image
def trimImage(im):
    heightImage = []
    cropImage = []
    pixels = toMatrix(im, list(im.getdata()))

    for x in range(len(pixels)):
        if sum(pixels[x]) != 0:
            heightImage.append(pixels[x])

    pixels = np.asarray(heightImage)
    pixels = pixels.T

    for x in range(len(pixels)):
        if sum(pixels[x]) != 0:
            cropImage.append(pixels[x])

    cropImage = np.asarray(cropImage)
    cropImage = np.rot90(cropImage, 0).T
    im = Image.fromarray(cropImage)
    return im


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

    pixels = toGreyScale(pixels)

    pixels = toMatrix(im, pixels)

    original = Image.fromarray(pixels)
    width, height = original.size
    original = original.crop((30, 0, width, height))
    original.show()
    #image is now black and white.
    pixelStartPos = 0
    lines = []

    while True:
        line, newPixelStartPos = getLine(original, pixelStartPos)
        pixelStartPos = newPixelStartPos + pixelStartPos
        if line == 0:
            break
        elif line == 1:
            pass
        else:
            lines.append("para")
            lines.append(line)


    splitImages = []
    for line in lines:
        if line != "para":
            pixelStartPos = 0

            while True:
                im, newPixelStartPos = splitImage(line, pixelStartPos)

                pixelStartPos = newPixelStartPos + pixelStartPos
                if newPixelStartPos > 300:
                    splitImages.append("space")
                if im == 0:
                    break
                elif im == 1:
                    pass
                else:
                    splitImages.append(im)
        else:
            splitImages.append("para")

#splitImage.append(Image.fromarray(np.array([0,0])))

    images = []
    for image in splitImages:
        if image == "space":
            images.append("space")
        elif image == "para":
            images.append("para")
        else:
            images.append(trimImage(image))


    #images are cropped to remove black rows and columns
    try:
        for x in range(len(images)):
            if images[x] != "space" and images[x] != "para":
                rows, cols = images[x].size
                if rows > cols:
                    factor = 20.0/rows
                    rows = 20
                    cols = int(round(cols*factor))
                    images[x] = images[x].resize((rows, cols))
                else:
                    factor = 20.0/cols
                    cols = 20
                    rows = int(round(rows*factor))
                    images[x] = images[x].resize((rows, cols))


                colsPadding = (int(math.ceil((28-cols)/2.0)), int(math.floor((28-cols)/2.0)))
                rowsPadding = (int(math.ceil((28-rows)/2.0)), int(math.floor((28-rows)/2.0)))

                pixelArray = np.lib.pad(images[x], (colsPadding, rowsPadding), 'constant')
                images[x] = Image.fromarray(pixelArray)
                shiftTuple = getCenterOfMassShift(pixelArray)
                shiftedImage = shift(images[x], shiftTuple)
                images[x] = shiftedImage

        # for x in range(len(images)):
        #     images[x].save('imageRecognition' + '/' + photoFolderName + '/' + 'formattedImages' +
        #         '/' + 'formattedImage' + str(x) + '.png')
        imagesPixels = []
        for image in images:
            if image == "space":
                imagesPixels.append("space")
            elif image == "para":
                imagesPixels.append("para")
            else:
                # image.show()
                imagesPixels.append(list(image.getdata()))


        return imagesPixels

    except:
        print("the photo quality wasn't good enough to read try again...")

        #error
        return None














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

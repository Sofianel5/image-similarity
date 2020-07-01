from PIL import Image
import math
import operator
from functools import reduce
from numpy import average, linalg, dot

DEFAULT_SIZE = (128, 128)
def normalize_size(img, size=DEFAULT_SIZE):
    img.thumbnail(size, Image.ANTIALIAS)
    return img 

def similarity_histogram(file1, file2):
    img1 = Image.open(file1)
    img2 = Image.open(file2)
    
    img1 = normalize_size(img1)
    img2 = normalize_size(img2)

    h1 = img1.histogram()
    h2 = img2.histogram()

    return math.sqrt(reduce(operator.add, list(map(lambda a,b: (a-b)**2, h1, h2)))/len(h1)) 

IMAGES_DIR = "test_similarity_images/"
print("Via histogram:")
print("Should be small:", similarity_histogram(IMAGES_DIR+"img1_a.jpg", IMAGES_DIR+"img1_b.jpg"))
print("Should be large:", similarity_histogram(IMAGES_DIR+"img1_a.jpg", IMAGES_DIR+"img2_a.jpg"))
print("Should be small:", similarity_histogram(IMAGES_DIR+"img2_a.jpg", IMAGES_DIR+"img2_b.jpeg"))
print("----------------------")

def similarity_vectors(file1, file2):
    img1 = Image.open(file1).resize(DEFAULT_SIZE)
    img2 = Image.open(file2).resize(DEFAULT_SIZE)
    images = [img1, img2]
    vectors = []
    norms = []
    for image in images:
        vector = [average(pixel) for pixel in image.getdata()]
        vectors.append(vector)
        norms.append(linalg.norm(vector,2))
    a,b=vectors 
    a_n,b_n = norms
    return dot(a/a_n, b/b_n)

print("Via pixel vectors")
print("Should be small:", similarity_vectors(IMAGES_DIR+"img1_a.jpg", IMAGES_DIR+"img1_b.jpg"))
print("Should be large:", similarity_vectors(IMAGES_DIR+"img1_a.jpg", IMAGES_DIR+"img2_a.jpg"))
print("Should be small:", similarity_vectors(IMAGES_DIR+"img2_a.jpg", IMAGES_DIR+"img2_b.jpeg"))


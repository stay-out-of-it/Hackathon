import numpy

from io import BytesIO
from mnist.loader import MNIST
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image


def classifier():
    mndata = MNIST("src/data/")
    images, labels = mndata.load_training()

    clf = KNeighborsClassifier()

    train_x = images[:100]
    train_y = labels[:100]

    clf.fit(train_x, train_y)

    return clf


def get_int_from_image(response, clf):
    pic = Image.open(BytesIO(response.content))
    pix = numpy.array(pic)

    q = [[j for i in pix for j in i]]
    predict = clf.predict(q)[0]

    return predict

from mnist.loader import MNIST
from PIL import Image


mndata = MNIST('src/data/')
images, labels = mndata.load_training()

i = 4
image, label = images[i], labels[i]

output = Image.new("L", (28, 28))
output.putdata(image)
output.save("output.png")

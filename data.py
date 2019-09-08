from mnist.loader import MNIST
from PIL import Image, ImageDraw

# Load dataset
mndata = MNIST('./data/')
images, labels = mndata.load_training()

# Pick the fifth image from the dataset (it's a 9)
i = 4
image, label = images[i], labels[i]

# Print the image
output = Image.new("L", (28, 28))
output.putdata(image)
output.save("output.png")

# Print label
print(label)

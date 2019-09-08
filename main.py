import requests

from flask import Flask

from .data1 import classifier, get_int_from_image


clf = classifier()
app = Flask(__name__)


@app.route('/image_to_int')
def hello(url):
    response = requests.get(url)
    predict = get_int_from_image(response, clf)
    return predict


if __name__ == '__main__':
    app.run()

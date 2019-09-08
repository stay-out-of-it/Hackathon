import requests as reqv

from flask import Flask, request

from src.data1 import classifier, get_int_from_image


clf = classifier()
app = Flask(__name__)


@app.route('/image_to_int', methods=['GET'])
def hello():
    url = request.args['url']
    response = reqv.get(url)
    predict = get_int_from_image(response, clf)
    return {'predict': int(predict)}


if __name__ == '__main__':
    app.run()

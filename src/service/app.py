from flask import Flask, request, jsonify
from flask_cors import CORS

from src.service.service import GeneratorService

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
CORS(app)

service = GeneratorService()



@app.route('/hello')
def home():
    return jsonify({'msg': 'hello! :)'})


@app.route('/faces')
def getFaces():
    ids = service.get_ids()
    return jsonify({'ids': ids})


@app.route('/faces', methods=['POST'])
def generateFaces():
    amount = request.args.get('amount')
    id = request.args.get('id')
    id1 = request.args.get('id1')
    id2 = request.args.get('id2')
    speed = request.args.get('speed')

    if id is not None and amount is None and id1 is None:
        id = service.generate_face(int(id))
        return jsonify({'id': id})

    if amount is not None and id is None and id1 is None:
        ids = service.generate_random_images(int(amount))
        return jsonify({'ids': ids})

    if amount is not None and id1 is not None:
        id1 = int(id1)
        amount = int(amount)

        if id2 is not None:
            id2 = int(id2)
        if speed is not None:
            speed = float(speed)
        
        service.generate_transition(id1, id2, amount, speed)
        return jsonify({'msg': "OK", 'code': 200})

    return jsonify({'msg': "Bad request", 'code': 404})


if __name__ == '__main__':
    app.run('0.0.0.0', 5000)

from flask import Flask, request, jsonify

import service

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


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

    if amount is None and id is not None:
        ids = service.generate_face(int(id))
        return jsonify({'ids': ids})

    if id1 is None:
        ids = service.generate_random_images(int(amount))
        return jsonify({'ids': ids})

    if id1 is not None and amount is not None and speed is not None:
        id1 = int(id1)
        if id2 is not None:
            id2 = int(id2)
        speed = float(speed)
        amount = int(amount)
        service.generate_transition(id1, id2, amount, speed)
        return 200

    return "Bad request", 404


if __name__ == '__main__':
    app.run('0.0.0.0', 5000)


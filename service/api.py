from flask import Flask, request, jsonify

from service.service import GeneratorService

generatorService = GeneratorService()

app = Flask(__name__)


@app.route('/hello')
def home():
    return jsonify({'msg': 'hello! :)'})


@app.route('/faces')
def getFaces():
    ids = generatorService.get_ids()
    return jsonify({'ids': ids})


@app.route('/faces', methods=['POST'])
def generateFaces():

    amount = request.args.get('amount')
    if amount is not None:
        ids = generatorService.generate_random_images(int(amount))
        return jsonify({'ids': ids})

    id1 = request.args.get('id1')
    id2 = request.args.get('id2')
    percentage = request.args.get('percentage')
    if id1 is not None and percentage is not None:
        id1 = int(id1)
        if id2 is not None:
            id2 = int(id2)
        percentage = float(percentage)
        generatorService.generate_transition(id1, id2, percentage)
        return 200

    return "Bad request", 404


if __name__ == '__main__':
    app.run('0.0.0.0', 5000)

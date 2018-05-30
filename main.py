import os

from Classify import classifier
from flask import Flask, jsonify, request

SITE_ROOT = os.path.realpath(os.path.dirname(__file__))

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({'Hello':'World'})

@app.route('/api/v1/survival', methods=['GET'])
def survival():
    if 'name' in request.args:
        csv_url = os.path.join(SITE_ROOT, 'data', 'train.csv')
        query = request.args['name']
        training_data = classifier.get_df(csv_url)
        training_data = classifier.get_clean_data(training_data)
        result = classifier.get_survival(training_data, query)
        return jsonify(result)
    else:
        return jsonify({'error': 'Please specify a name parameter'})
    return jsonify({'name': query})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

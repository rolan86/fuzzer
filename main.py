from flask import Flask, jsonify, request


app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({'Hello':'World'})

@app.route('/survival', methods=['GET'])
def survival():
    if 'name' in request.args:
        query = request.args['name']
    else:
        return jsonify({'error': 'Please specify a name parameter'})
    return jsonify({'name': query})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

import logging

import flask
from flasgger import Swagger
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from bert.train import load_model, disaster

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# NOTE this import needs to happen after the logger is configured


# Initialize the Flask application
application = Flask(__name__)

application.config['ALLOWED_EXTENSIONS'] = set(['pdf'])
application.config['CONTENT_TYPES'] = {"pdf": "application/pdf"}
application.config["Access-Control-Allow-Origin"] = "*"

CORS(application)

swagger = Swagger(application)

bert_layer, model = load_model(1)


def clienterror(error):
    resp = jsonify(error)
    resp.status_code = 400
    return resp


def notfound(error):
    resp = jsonify(error)
    resp.status_code = 404
    return resp


@application.route('/v1/tweet', methods=['POST'])
def tweet_classification():
    """Run tweet classification given text.
        ---
        parameters:
          - name: body
            in: body
            schema:
              id: text
              required:
                - text
              properties:
                text:
                  type: string
            description: the required text for POST method
            required: true
        definitions:
          Disaster-Related Tweets Response:
          Project:
            properties:
              status:
                type: string
              ml-result:
                type: object
        responses:
          40x:
            description: Client error
          200:
            description: Disaster-Related Tweets Response
            examples:
                          [
{
  "status": "success",
  "disaster": "1"
},
{
  "status": "error",
  "message": "Exception caught"
},
]
        """
    json_request = request.get_json()
    if not json_request:
        return Response("No json provided.", status=400)
    text = json_request['text']
    if text is None:
        return Response("No text provided.", status=400)
    else:
        label = disaster(text, bert_layer, model)
        return flask.jsonify({"status": "success", "label": label})


@application.route('/v1/tweet/categories', methods=['GET'])
def tweet_categories():
    """Possible tweet categories.
        ---
        definitions:
          CategoriestResponse:
          Project:
            properties:
              categories:
                type: object
        responses:
          40x:
            description: Client error
          200:
            description: Disaster-Related Tweet Classification Response
            examples:
                          [
{
  "categories": [1,2,3],
  "sentiment": "1"
}
]
        """
    return flask.jsonify({"categories": list([0, 1])})


if __name__ == '__main__':
    application.run(debug=True, use_reloader=True)

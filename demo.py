import json

from flask import Flask, jsonify
import tensorflow as tf
from flask import render_template
from flask import request

from run import Run
from data import Data


def demo(config):
    app = Flask(__name__)

    with tf.device("/cpu:0"):
        demo_run = Run(config)

        supervisor = tf.train.Supervisor(logdir=config.out_dir,
                                         save_model_secs=config.save_model_secs, summary_op=None)
        coord = tf.train.Coordinator()

    @app.route("/")
    def index():
        return render_template('index.html')

    @app.route("/_get_answer")
    def get_answer():
        context = request.args.get('context', "", type=str)
        ques = request.args.get('ques', "", type=str)
        squad = create_squad(context, ques)
        with open(config.squad_path, 'w') as fp:
            json.dump(squad, fp)
        demo_run.data.load()

        with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            demo_run.start_queue(sess, coord)
            e = demo_run.lap(supervisor, sess)
            print(e)

            # coord.request_stop()
            # demo_run.join_queue(coord)
            answer = next(iter(e.get_answers().values()))
            print(answer)
            return jsonify(answer=answer)

    app.run(port=config.port, host=config.host)


def create_squad(context, ques):
    squad = {'data': [{'paragraphs': [{'context': context, 'qas': [{'id': '0', 'question': ques, 'answers': [{'answer_start': 0, 'text': context[:1]}]}]}]}]}
    return squad

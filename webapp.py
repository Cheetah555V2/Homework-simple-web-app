import random
import time
import os
import hashlib

from flask import Flask, render_template, request

from calcs import plot_expression

# Docs and examples for Flask: https://flask.palletsprojects.com/en/stable/
app = Flask(__name__)  # To run, use flask --app webapp run --debug

@app.route("/")  # http://127.0.0.1:5000/
def main_page():
    plot_file = ''
    version = ''
    user_input = request.args.get('func_expr', '').strip()

    min_interval = request.args.get('min_interval', type=float)
    max_interval = request.args.get('max_interval', type=float)

    color = request.args.get('color', 'blue', type=str)

    print(color, color[0:2])

    # set sensible defaults if not provided
    if min_interval is None:
        min_interval = -2.0
    if max_interval is None:
        max_interval = 2.0

    if user_input != '':
        os.makedirs('static', exist_ok=True) # make sure that dir static exist

        unhash_text = user_input + '&' + str(min_interval) + '&' + str(max_interval) + '&' + str(color)
        h = hashlib.sha256(unhash_text.encode('utf-8')).hexdigest()[:12]
        plot_file = f"plot_{h}.png"
        plot_path = os.path.join('static', plot_file)

        try:
            plot_expression(user_input, min_interval, max_interval, color, plot_path)
            version = str(int(time.time() * 1000))  # cache-buster for the browser
        except Exception:
            plot_file = ''
            version = ''

    return render_template('plot_func.html',
                           func_expr=user_input,
                           plot_file=plot_file,
                           version=version,
                           min_interval=min_interval,
                           max_interval=max_interval,
                           color=color)

@app.route("/test")  # http://127.0.0.1:5000/test
def test_route():
    x = random.randint(0, 10)

    return render_template('main_page.html', lucky_num=x)

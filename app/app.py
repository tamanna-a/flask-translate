from flask import Flask, render_template, url_for, redirect, request
import torch
import torch.nn as nn
import unicodedata
import string
import json
import unicodedata
import re
import random
import pickle
from process import normalizeEnglish

from model import predict

app = Flask(__name__)

'''
# Load Language Classes as pickle objects
class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "app"
        return super().find_class(module, name)

pickle_eng_in = open("resources/eng.pickle", "rb")
unpickler = MyCustomUnpickler(pickle_eng_in)
input_lang = unpickler.load()
print(input_lang.name)

pickle_spa_in = open("resources/spa.pickle", "rb")
unpickler = MyCustomUnpickler(pickle_spa_in)
output_lang = unpickler.load()
print(output_lang.name)

#define model variables
hidden_size = 256
MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#load models
encoder = EncoderRNN(input_lang.n_words, hidden_size)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1)
encoder.load_state_dict(torch.load('resources/encoder_3.h5', map_location=torch.device('cpu')))
decoder.load_state_dict(torch.load('resources/decoder_3.h5', map_location=torch.device('cpu')))
'''

@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template('translate.html')


@app.route("/hello")
def hello():
    return render_template('hello.html')


@app.route("/about", methods = ['GET'])
def about():
    return render_template('about.html')


@app.route("/feedback", methods = ['GET'])
def feedback():
    return render_template('feedback.html')


@app.route("/<usr>")
def user(usr):
    return f"<h1> {usr} </h1>"


def format_output(l):
    s = ' '.join(l[:-1]) #dont include EOS token
    s = re.sub(r'\s([?.!"](?:\s|$))', r'\1', s) #remove space before punctuation
    return s


@app.route("/translate", methods =['POST', 'GET'])
def translate():
    if request.method == 'POST':
        text = request.form['user_input']
        processed_text = normalizeEnglish(text)
        output = predict(processed_text)
        formatted = format_output(output)
        return render_template('result.html', result=formatted)
    else:
        return render_template('translate.html')



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5102)





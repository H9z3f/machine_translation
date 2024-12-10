from flask import Flask, request, jsonify, render_template
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string

# Загрузка модели и токенизаторов
model = load_model('models/seq2seq_model.keras')

with open('models/eng_tokenizer.pkl', 'rb') as f:
    eng_tokenizer = pickle.load(f)

with open('models/deu_tokenizer.pkl', 'rb') as f:
    deu_tokenizer = pickle.load(f)

# Параметры модели
deu_length = 8
eng_length = 8

# Функция для предсказания перевода
def predict_sentence(sentence):
    # Предобработка предложения
    sentence = sentence.lower().translate(str.maketrans('', '', string.punctuation))
    seq = deu_tokenizer.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=deu_length, padding='post')

    # Предсказание
    preds = model.predict(seq)
    pred_indices = preds.argmax(axis=-1)[0]

    # Преобразование индексов в текст
    words = []
    for idx in pred_indices:
        word = None
        for w, i in eng_tokenizer.word_index.items():
            if i == idx:
                word = w
                break
        if word:
            words.append(word)
    return ' '.join(words)

# Создание Flask-приложения
app = Flask(__name__, static_folder='templates')

# Главная страница с формой ввода
@app.route('/')
def home():
    return render_template('index.html')

# API для перевода
@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    german_sentence = data.get('sentence', '')
    if not german_sentence:
        return jsonify({'error': 'No input sentence provided'}), 400
    
    english_translation = predict_sentence(german_sentence)
    return jsonify({'german': german_sentence, 'english': english_translation})

if __name__ == '__main__':
    app.run(debug=True)

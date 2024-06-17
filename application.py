from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from werkzeug.utils import secure_filename
import os
from src.pipelines.prediction_pipeline import predict
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Создаем папку для загрузки файлов, если ее нет
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        df = pd.read_csv(file_path)

        predictions = predict(df)
        final = pd.concat([df['ID'], pd.Series(predictions).map(
            {0: 'ok', 1: 'fraud'})], axis=1)

        return render_template('result.html', tables=[df.to_html(classes='data'), final.to_html(classes='data')], titles=['Загруженные данные', 'Предсказания'])

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)

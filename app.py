from flask import Flask, render_template, request
import pandas as pd
import os
from sklearn.cluster import KMeans
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded.")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No selected file.")

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            return render_template('index.html', error="Unsupported file format.")

        if not set(['age', 'price']).issubset(df.columns):
            return render_template('index.html', error="Missing required columns: age, price")

        if 'gender' in df.columns:
            df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

        X = df[['age', 'price']].values
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['cluster'] = kmeans.fit_predict(X)

        centers = kmeans.cluster_centers_
        centers_list = [f"[{round(c[0], 2)}, {round(c[1], 2)}]" for c in centers]

        table_html = df.to_html(classes='data', header="true", index=False)
        return render_template('index.html', tables=table_html, centers=centers_list)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__)
output_dir = "../data/output/"

@app.route('/')
def index():
    # List all detected images
    images = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".jpg"):
                images.append(os.path.join(root, file))
    return render_template('index.html', images=images)

@app.route('/images/<path:filename>')
def images(filename):
    return send_from_directory(output_dir, filename)

if __name__ == '__main__':
    app.run(debug=True)

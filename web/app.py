from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate/realtime')
def translate_realtime():
    return render_template('translate.html')

@app.route('/translate/upload')
def translate_upload():
    return render_template('translate_upload.html')

@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')

if __name__ == '__main__':
    app.run(debug=True)

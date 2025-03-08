from flask import Flask, request, jsonify, render_template, redirect, g
import sqlite3

DATABASE = "curiosity_lab.db"
app = Flask(__name__)

def connect_db():
    rv = sqlite3.connect(DATABASE)
    rv.row_factory = sqlite3.Row
    return rv

def get_db():
    if not hasattr(g, 'sqlite_db'):
        g.sqlite_db = connect_db()
    return g.sqlite_db

@app.teardown_appcontext
def close_db(exception):
    """リクエスト終了時にデータベースを閉じる"""
    db = g.pop('sqlite_db', None)
    if db is not None:
        db.close()

def init_db():
    """データベースを初期化する"""
    with app.app_context():
        db = get_db()
        with app.open_resource('schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()

@app.route('/')
def index():

    return render_template('regist.html')

@app.route('/register', methods=['POST'])
def register():

    name = request.form.get('name')
    interest = request.form.get('interest')
    interest_text = request.form.get('interest_text')

    db = get_db()
    db.execute("INSERT INTO curiosity (name, interest, interest_text) VALUES (?, ?, ?)", (name, interest, interest_text))
    db.commit()

    return redirect('/')

if __name__ == '__main__':
    init_db()
    app.run(debug=True)


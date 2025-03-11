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

    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register():

    name = request.form.get('name') #ユーザー名
    dname = request.form.get('dname') #大分類名
    iname = request.form.get('iname') #小分類名
    dcode = request.form.get('dcode') #大分類コード
    icode = request.form.get('icode') #小分類コード
    interest_text = request.form.get('interest_text') #興味の詳細

    # デバッグ出力
    print(f"DEBUG: name={name}, dname={dname}, dcode={dcode}, iname={iname}, icode={icode}, interest_text={interest_text}")

    if not dname or not dcode:
        return "Error: 大分類が選択されていません", 400
    if not iname or not icode:
        return "Error: 小分類が選択されていません", 400

    db = get_db()
    db.execute("INSERT INTO curiosity (name, dname, iname, dcode, icode, interest_text) VALUES (?, ?, ?, ? ,? ,?)",
                (name, dname, iname, dcode, icode, interest_text))
    db.commit()

    #return jsonify({"message": "登録が完了しました"}), 200

    return redirect('/')

if __name__ == '__main__':
    init_db()
    app.run(debug=True)

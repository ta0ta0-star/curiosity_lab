from flask import Flask
from flask import request, jsonify, render_template, redirect, g
import sqlite3
DATABASE="curiosity_lab.db"

app = Flask(__name__)

if __name__ == '__main__':
    app.run(debug=True)

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


@app.route('/')
def index():

    return render_template('regist.html')

@app.route('/register', methods=['POST'])
def register():
    if request.method == 'POST':
        #画面からの登録情報の取得
        name = request.form.get('name')
        interest = request.form.get('interest')
        interest_text = request.form.get('interest_text')
        #data = request.form
        db = get_db()
        db.execute("insert into curiosity (name, interest, interest_text) values(?, ?, ?)" ,[name, interest, interest_text])
        db.commit()
        return redirect('/')
    

    # ここでデータを処理します（例：データベースに保存）

    return jsonify({"message": f"登録が完了しました。名前: {name}, 興味: {interest}, 詳細: {interest_text}"}), 200


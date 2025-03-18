from flask import Flask, request, jsonify, render_template, redirect, g, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import functools
import sqlite3

DATABASE_CURIOSITY = "curiosity_lab.db"
DATABASE_USERS = "users.db"
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

def connect_db(database):
    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row
    return conn

def get_db(database):
    # データベースごとに異なるキーを使用
    db_key = f'sqlite_db_{database}'
    if not hasattr(g, db_key):
        conn = connect_db(database)
        setattr(g, db_key, conn)
    return getattr(g, db_key)

@app.teardown_appcontext
def close_db(exception):
    """リクエスト終了時にデータベースを閉じる"""
    # 両方のデータベース接続を閉じる
    for key in list(g.__dict__.keys()):
        if key.startswith('sqlite_db_'):
            db = g.pop(key, None)
            if db is not None:
                db.close()

def init_db():
    """データベースを初期化する"""
    with app.app_context():
        db_curi = get_db(DATABASE_CURIOSITY)
        db_users = get_db(DATABASE_USERS)
        # with app.open_resource('schema.sql', mode='r') as f:
        #     db.cursor().executescript(f.read())

        # curiosity_lab.db の初期化
        db_curi.execute('''
            CREATE TABLE IF NOT EXISTS curiosity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dname TEXT NOT NULL,
                iname TEXT NOT NULL,
                dcode TEXT NOT NULL,
                icode TEXT NOT NULL,
                interest_text TEXT
            );
        ''')
        db_curi.commit()
        # users.db の初期化
        db_users.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                gender TEXT NOT NULL,
                birthday TEXT NOT NULL,
                email TEXT NOT NULL,
                phone_number TEXT NOT NULL,
                school TEXT,
                dname TEXT NOT NULL,
                iname TEXT NOT NULL,
                dcode TEXT NOT NULL,
                icode TEXT NOT NULL,
                interest_text TEXT
            );
        ''')
        db_users.commit()


@app.route('/')
def index():

    return render_template('index.html')

# 興味情報の登録
@app.route('/register', methods=['POST'])
def register_curiosity():
    dname = request.form.get('dname')
    iname = request.form.get('iname')
    dcode = request.form.get('dcode')
    icode = request.form.get('icode')
    interest_text = request.form.get('interest_text')

    # デバッグ出力
    print(f"DEBUG: dname={dname}, dcode={dcode}, iname={iname}, icode={icode}, interest_text={interest_text}")

    if not dname or not dcode:
        return "Error: 大分類が選択されていません", 400
    if not iname or not icode:
        return "Error: 小分類が選択されていません", 400

    db = get_db(DATABASE_CURIOSITY)
    db.execute("INSERT INTO curiosity (dname, iname, dcode, icode, interest_text) VALUES (?, ?, ?, ?, ?)",
               (dname, iname, dcode, icode, interest_text))
    db.commit()

    #return redirect('/')

# 個人情報の登録ページ
@app.route('/user_register', methods=['GET', 'POST'])
def register_user():
    if request.method == 'GET':
        return render_template('register.html')

    # フォームデータ取得
    name = request.form.get('name')
    gender = request.form.get('gender')
    birthday = request.form.get('birthday')
    email = request.form.get('email')
    phone_number = request.form.get('phone_number')
    school = request.form.get('school')
    dname = request.form.get('dname')
    iname = request.form.get('iname')
    dcode = request.form.get('dcode')
    icode = request.form.get('icode')
    interest_text = request.form.get('interest_text')

    # デバッグ出力
    print(f"DEBUG: name={name}, gender={gender}, birthday={birthday}, email={email}, phone={phone_number}, dname={dname}, iname={iname}, dcode={dcode}, icode={icode}, interest_text={interest_text}")

    if not name or not gender or not birthday or not email or not phone_number or not dname or not iname or not dcode or not icode:
        return jsonify({"message": "入力が不足しています"}), 400

    db = get_db(DATABASE_USERS)
    db.execute('''
        INSERT INTO users (name, gender, birthday, email, phone_number, school, dname, iname, dcode, icode, interest_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (name, gender, birthday, email, phone_number, school, dname, iname, dcode, icode, interest_text))
    db.commit()

    # return jsonify({"message": "登録が完了しました"}), 200

    return redirect(url_for('index'))

# 登録されたデータの一覧を表示
@app.route('/users')
def list_users():
    db = get_db(DATABASE_USERS)
    users = db.execute("SELECT * FROM users").fetchall()
    return render_template('users.html', users=users)

# ユーザー編集ページ（編集フォームを表示）
@app.route("/<int:id>/edit", methods=['GET'])
def edit(id):
    user = get_db(DATABASE_USERS).execute(
        "SELECT id, name, gender, birthday, email, phone_number, school, dname, iname, dcode, icode, interest_text from users where id=?", (id,)
    ).fetchone()
    return render_template('edit.html', user=user)

# ユーザー更新処理
@app.route("/<int:id>/update", methods=['POST'])
def update(id):
    # フォームデータ取得
    name = request.form.get('name')
    gender = request.form.get('gender')
    birthday = request.form.get('birthday')
    email = request.form.get('email')
    phone_number = request.form.get('phone_number')
    school = request.form.get('school')
    dname = request.form.get('dname')
    iname = request.form.get('iname')
    dcode = request.form.get('dcode')
    icode = request.form.get('icode')
    interest_text = request.form.get('interest_text')

    # 入力チェック
    if not name or not gender or not birthday or not email or not phone_number or not dname or not iname or not dcode or not icode:
        return jsonify({"message": "入力が不足しています"}), 400

    # データベース更新
    db = get_db(DATABASE_USERS)
    db.execute('''
        UPDATE users
        SET name=?, gender=?, birthday=?, email=?, phone_number=?, school=?, 
            dname=?, iname=?, dcode=?, icode=?, interest_text=?
        WHERE id=?
    ''', (name, gender, birthday, email, phone_number, school, dname, iname, dcode, icode, interest_text, id))
    db.commit()

    return redirect(url_for('list_users'))

# ユーザー削除処理
@app.route('/delete/<int:id>')
def delete(id):
    db = get_db(DATABASE_USERS)
    db.execute("DELETE FROM users WHERE id = ?", (id,))
    db.commit()
    return redirect(url_for('list_users'))

# ローディング画面
@app.route('/load')
def load():
    return render_template('load.html')


if __name__ == '__main__':
    init_db()
    app.run(debug=True)

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
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
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
    return jsonify({"message": "登録が完了しました"}), 200


# ログイン必須デコレータ
def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return view(**kwargs)
    return wrapped_view

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
    password = request.form.get('password')  # パスワードを取得
    phone_number = request.form.get('phone_number')
    school = request.form.get('school')
    dname = request.form.get('dname')
    iname = request.form.get('iname')
    dcode = request.form.get('dcode')
    icode = request.form.get('icode')
    interest_text = request.form.get('interest_text')

    # デバッグ出力
    print(f"DEBUG: name={name}, gender={gender}, birthday={birthday}, email={email}, password={password}, phone={phone_number}, dname={dname}, iname={iname}, dcode={dcode}, icode={icode}, interest_text={interest_text}")

    if not name or not gender or not birthday or not email or not password or not phone_number or not dname or not iname or not dcode or not icode:
        flash('入力が不足しています')
        return render_template('register.html')

    db = get_db(DATABASE_USERS)

    # メールアドレスの重複チェック
    if db.execute("SELECT id FROM users WHERE email = ?",(email,)).fetchone() is not None:
        flash('そのメールアドレスは既に登録されています')
        return render_template('register.html')

    password_hash = generate_password_hash(password)  # パスワードをハッシュ化

    try:
        db.execute('''
            INSERT INTO users (name, gender, birthday, email, password_hash, phone_number, school, dname, iname, dcode, icode, interest_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (name, gender, birthday, email, password_hash, phone_number, school, dname, iname, dcode, icode, interest_text))
        db.commit()
        flash('ユーザー登録が完了しました。ログインしてください。')
        return redirect(url_for('login'))
    except Exception as e:
        print(f"エラー: {e}")
        flash('登録中にエラーが発生しました。')
        return render_template('register.html')

# ログイン処理
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')

    email = request.form.get('email')
    password = request.form.get('password')

    db = get_db(DATABASE_USERS)
    user = db.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()

    error = None
    if user is None:
        error = 'メールアドレスが正しくありません。'
    elif not check_password_hash(user['password_hash'], password):
        error = 'パスワードが正しくありません。'

    if error is None:
    # セッションをクリアして新しいユーザーIDを設定
        session.clear()
        session['user_id'] = user['id']
        session['user_name'] = user['name']
        flash('ログインしました')
        return redirect(url_for('index'))
    
    flash(error)
    return render_template('login.html')

# ログアウト処理
@app.route('/logout')
def logout():
    session.clear()
    flash('ログアウトしました')
    return redirect(url_for('index'))

# 現在のユーザー情報を獲得
@app.before_request
def load_logged_in_user():
    user_id = session.get('user_id')

    if user_id is None:
        g.user = None
    else:
        g.user = get_db(DATABASE_USERS).execute(
            'SELECT * FROM users WHERE id = ?', (user_id,)
        ).fetchone()

# 登録されたデータの一覧を表示
@app.route('/users')
@login_required
def list_users():
    db = get_db(DATABASE_USERS)
    users = db.execute("SELECT * FROM users").fetchall()
    return render_template('users.html', users=users)

# ユーザー編集ページ（編集フォームを表示）
@app.route("/<int:id>/edit", methods=['GET'])
@login_required
def edit(id):
    user = get_db(DATABASE_USERS).execute(
        "SELECT id, name, gender, birthday, email, phone_number, school, dname, iname, dcode, icode, interest_text from users where id=?", (id,)
    ).fetchone()
    return render_template('edit.html', user=user)

# ユーザー更新処理
@app.route("/<int:id>/update", methods=['POST'])
@login_required
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
    user_id = session.get('user_id')
    user = get_db(DATABASE_USERS).execute(
        "SELECT * FROM users WHERE id = ?", (user_id,)
    ).fetchone()

    return render_template('profile.html', user=user)


# ユーザー削除処理
@app.route('/delete/<int:id>')
@login_required
def delete(id):
    db = get_db(DATABASE_USERS)
    db.execute("DELETE FROM users WHERE id = ?", (id,))
    db.commit()
    return redirect(url_for('list_users'))

# ローディング画面
@app.route('/load')
def load():
    return render_template('load.html')

# マイページ表示
@app.route('/profile')
@login_required
def profile():
    # 現在ログインしているユーザーの情報を取得
    user_id = session.get('user_id')
    user = get_db(DATABASE_USERS).execute(
        "SELECT * FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    
    if user is None:
        flash('ユーザー情報が見つかりません')
        return redirect(url_for('index'))
    
    return render_template('profile.html', user=user)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)

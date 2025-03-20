import functools
import sqlite3
from flask import Flask, request, jsonify, render_template, redirect, g, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
import re
import os
import langdetect
from gensim.models import FastText
import fasttext
import fasttext.util
import pickle
import sys
sys.path.append('path_to_the_directory_containing_my_module')
from my_module import get_text_vector



DATABASE_CURIOSITY = "curiosity_lab.db"
DATABASE_USERS = "users.db"
DATABASE_RE = "researcher_matching.db"
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# NLTKのデータをダウンロード
nltk.download('stopwords')
nltk.download('punkt')

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
        db_re = get_db(DATABASE_RE)
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
        # researcher_matching.db の初期化
        db_re.execute('''
            CREATE TABLE IF NOT EXISTS interests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                dname TEXT NOT NULL,
                iname TEXT NOT NULL,
                dcode TEXT NOT NULL,
                icode TEXT NOT NULL,
                interest_text TEXT,
                created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            );
        ''')
        db_re.commit()

def parse_research_area(area_str):
    """
    'dname/cname/' や 'dname/cname' の形式の文字列をリストに変換する
    """
    if not isinstance(area_str, str):
        return []

    areas = [x.strip("/") for x in area_str.split(", ")]
    parsed_areas = [area.split("/") for area in areas]
    return [{"dname": a[0], "cname": a[1]} for a in parsed_areas if len(a) == 2]

def is_english(text):
    """テキストが英語かどうかを判定する"""
    if not isinstance(text, str) or not text.strip():
        return False

    try:
        return langdetect.detect(text) == 'en'
    except:
        # 言語検出に失敗した場合は英語ではないと判断
        return False

@app.route('/')
def index():
    return render_template('index.html')

# 興味情報の登録
@app.route('/register', methods=['POST'])
def register_curiosity():
    if request.method == 'POST':

        dname = request.form.get('dname')
        iname = request.form.get('iname')
        dcode = request.form.get('dcode')
        icode = request.form.get('icode')
        interest_text = request.form.get('interest_text', '')

        # セッションに保存
        session['user_interests'] = {
            'dname': dname,
            'iname': iname,
            'dcode': dcode,
            'icode': icode,
            'interest_text': interest_text
        }

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

    return jsonify({"status": "success"})


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


# 研究者データのCSV
RESEARCHER_CSV = "researcher_data1.csv"
VECTOR_CACHE_FILE = "vector_cache.pkl"

# 研究者データのCSV
RESEARCHER_CSV = "researcher_data1.csv"
VECTOR_CACHE_FILE = "vector_cache.pkl"

def load_fasttext_models():
    # 日本語モデルをロード
    ft_model_ja = fasttext.load_model("cc.ja.300.bin")
    # 英語モデルをロード
    ft_model_en = fasttext.load_model("cc.en.300.bin")
    return ft_model_ja, ft_model_en

# 研究者データの事前処理とベクトル計算（キャッシュ機能付き）
def preprocess_and_vectorize_researchers():
    # キャッシュファイルが存在する場合はロード
    if os.path.exists(VECTOR_CACHE_FILE):
        try:
            print("キャッシュからベクトルをロード中...")
            with open(VECTOR_CACHE_FILE, 'rb') as f:
                cache_data = pickle.load(f)
                # キャッシュデータの検証
                if 'researchers_df' in cache_data and 'researcher_vectors' in cache_data:
                    print("キャッシュからのロード成功")
                    return cache_data['researchers_df'], cache_data['researcher_vectors']
        except Exception as e:
            print(f"キャッシュロードエラー: {e}")
    
    print("研究者データを前処理中...")
    df = pd.read_csv(RESEARCHER_CSV)
    
    # 前処理 - 欠損値を空文字列に置き換え（一括処理）
    text_columns = ['name', 'research_experience', 'association_memberships', 'research_areas', 'research_interests', 'published_papers']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna('')
        else:
            df[col] = ''  # カラムが存在しない場合は空の列を作成
    
    # 研究分野の分類情報を抽出
    df['extracted_areas'] = df['research_areas'].apply(
        lambda x: parse_research_area(x) if isinstance(x, str) else []
    )
    
    # 研究者のテキストデータを結合（ベクトル化のため）
    df["combined_text"] = df.apply(
        lambda row: f"{row['name']} {row['research_experience']} {row['research_areas']} {row['research_interests']} {row['published_papers']}", 
        axis=1
    )
    
    # FastTextモデルをロード
    ft_model_ja, ft_model_en = load_fasttext_models()
    
    # 研究者データのベクトルを一括計算（並列処理可能）
    print("研究者ベクトルを計算中...")
    
    # 並列処理のためにPandasのapply関数を使用
    from concurrent.futures import ThreadPoolExecutor
    
    def process_text(text):
        return get_text_vector(text, ft_model_ja, ft_model_en)
    
    # ThreadPoolExecutorを使用して並列処理
    with ThreadPoolExecutor(max_workers=4) as executor:
        researcher_vectors = list(executor.map(process_text, df['combined_text']))
    
    # NumPy配列に変換して高速化
    researcher_vectors = np.array(researcher_vectors)
    
    # キャッシュに保存
    cache_data = {
        'researchers_df': df,
        'researcher_vectors': researcher_vectors
    }
    
    with open(VECTOR_CACHE_FILE, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"研究者データ: {len(df)}件のベクトル化完了")
    return df, researcher_vectors

# 研究者データとのマッチング処理（高速化版）
def match_researchers(user_interests):
    try:
        # 研究者データとベクトルをロード（キャッシュから）
        researchers_df, researcher_vectors = preprocess_and_vectorize_researchers()
        
        # FastTextモデルをロード
        ft_model_ja, ft_model_en = load_fasttext_models()
        
        # ユーザーの興味・関心情報を取得
        user_dname = user_interests.get('dname', '')
        user_iname = user_interests.get('iname', '')
        user_interest_text = user_interests.get('interest_text', '')
        
        # ユーザーの興味・関心をベクトル化
        user_text = f"{user_dname} {user_iname} {user_interest_text}"
        user_vector = get_text_vector(user_text, ft_model_ja, ft_model_en)
        
        print(f"ユーザーテキスト: {user_text}")
        
        # NumPyを使った高速なコサイン類似度計算
        # 1. ゼロベクトルのインデックスを特定
        zero_vectors = np.all(researcher_vectors == 0, axis=1)
        
        # 2. ユーザーベクトルがゼロベクトルかチェック
        if np.all(user_vector == 0):
            similarities = np.zeros(len(researcher_vectors))
        else:
            # 3. 非ゼロベクトルに対してコサイン類似度を計算
            # ドット積を計算
            dot_products = np.dot(researcher_vectors, user_vector)
            
            # ノルムを計算
            researcher_norms = np.linalg.norm(researcher_vectors, axis=1)
            user_norm = np.linalg.norm(user_vector)
            
            # コサイン類似度を計算（ゼロ除算を避ける）
            with np.errstate(divide='ignore', invalid='ignore'):
                similarities = dot_products / (researcher_norms * user_norm)
            
            # NaNや無限大を0に置き換え
            similarities = np.nan_to_num(similarities)
            
            # ゼロベクトルの類似度を0に設定
            similarities[zero_vectors] = 0
        
        # 類似度をDataFrameに追加
        researchers_df['similarity'] = similarities
        
        # 類似度の統計情報を出力
        print(f"類似度の最大値(重みづけなし): {np.max(similarities)}")
        print(f"類似度の最小値(重みづけなし): {np.min(similarities)}")
        print(f"類似度の平均値(重みづけなし): {np.mean(similarities)}")
        
        # 小分類の一致度に基づいて類似度を調整（ベクトル化操作）
        # 1. 大分類と小分類の一致を確認するための関数
        def check_matches(row):
            areas = row['extracted_areas']
            subcategory_match = False
            major_category_match = False
            
            for area in areas:
                if area.get('dname') == user_dname:
                    major_category_match = True
                    if area.get('cname') == user_iname:
                        subcategory_match = True
                        break
            
            return pd.Series([subcategory_match, major_category_match])
        
        # 2. 一括で一致チェック
        researchers_df[['subcategory_match', 'major_category_match']] = researchers_df.apply(check_matches, axis=1)
        
        # 3. 条件に基づいて類似度を調整（ベクトル化操作）
        conditions = [
            researchers_df['subcategory_match'],
            researchers_df['major_category_match'] & ~researchers_df['subcategory_match'],
            ~researchers_df['major_category_match'] & ~researchers_df['subcategory_match']
        ]
        
        choices = [
            researchers_df['similarity'] * 1.5,  # 小分類一致
            researchers_df['similarity'] * 0.8,  # 大分類のみ一致
            researchers_df['similarity'] * 0.5   # どちらも不一致
        ]
        
        researchers_df['adjusted_similarity'] = np.select(conditions, choices, default=researchers_df['similarity'])
        
        # 調整後の類似度でソート
        researchers_df = researchers_df.sort_values('adjusted_similarity', ascending=False)
        
        # 上位10件の研究者データを返す
        top_researchers = []
        for _, row in researchers_df.head(10).iterrows():
            researcher = row.to_dict()
            researcher['similarity'] = researcher['adjusted_similarity'] * 100  # パーセンテージに変換
            top_researchers.append(researcher)
        
        return top_researchers
    
    except Exception as e:
        print(f"マッチング処理中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        # エラーが発生した場合は空のリストを返す
        return []

# マッチング結果の表示
@app.route("/results")
def show_results():
    user_interests = session.get("user_interests", {})

    if not user_interests:
        return redirect(url_for("index"))

    matched_researchers = match_researchers(user_interests)

    return render_template("results.html", researchers=matched_researchers, interests=user_interests)

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

if __name__ == "__main__":
    if not os.path.exists(DATABASE_RE):
        init_db()
    app.run(debug=True)

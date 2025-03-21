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
from concurrent.futures import ThreadPoolExecutor
# import Mecab
from sentence_transformers import SentenceTransformer, util



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

CACHE_VERSION = "1.1"  # キャッシュのバージョン（コードを変更したら更新する）

def load_xlmr_model():
    print("XLM-R モデルをロード中...")
    # xlm-roberta-base is a good multilingual model that supports Japanese and English
    model = SentenceTransformer('xlm-roberta-base')
    print("✅ XLM-R モデルのロード成功")
    return model

# テキストをXLM-Rでベクトル化
def get_text_vector(text, model=None):
    if model is None:
        # Lazy loading of the model if not provided
        model = load_xlmr_model()

    if not text or not isinstance(text, str):
        # Return zero vector of appropriate size for empty text
        return np.zeros(768)  # XLM-R base model has 768 dimensions

    # XLM-R can handle raw text without tokenization
    return model.encode(text)

# 研究者データの処理とベクトル化
def preprocess_and_vectorize_researchers():
    # Check for cache
    if os.path.exists(VECTOR_CACHE_FILE):
        with open(VECTOR_CACHE_FILE, 'rb') as f:
            cache_data = pickle.load(f)
            if cache_data.get('version') == CACHE_VERSION:
                return cache_data['df'], cache_data['vectors']

    # Load researcher data
    df = pd.read_csv(RESEARCHER_CSV, encoding="utf-8")
    df.fillna('', inplace=True)
    
    # Combine text fields
    df["combined_text"] = df.apply(
        lambda row: f"{row['name']} {row['research_areas']} {row['research_interests']} {row['published_papers']}",
        axis=1
    )
    
    # Load XLM-R model
    model = load_xlmr_model()
    
    print("研究者ベクトルを計算中...")
    # Vectorize all researcher texts
    vectors = []
    batch_size = 32  # Process in batches to avoid memory issues
    
    for i in range(0, len(df), batch_size):
        batch_texts = df['combined_text'].iloc[i:i+batch_size].tolist()
        batch_vectors = model.encode(batch_texts)
        vectors.extend(batch_vectors)
    
    vectors = np.array(vectors)
    
    # Save to cache
    with open(VECTOR_CACHE_FILE, 'wb') as f:
        pickle.dump({'version': CACHE_VERSION, 'df': df, 'vectors': vectors}, f)
    
    return df, vectors

# マッチング処理
def match_researchers(user_interests):
    try:
        # Load researcher data and vectors
        researchers_df, researcher_vectors = preprocess_and_vectorize_researchers()
        
        # Get user interests
        user_dname = user_interests.get('dname', '')
        user_iname = user_interests.get('iname', '')
        user_interest_text = user_interests.get('interest_text', '')
        
        # Combine user text
        user_text = f"{user_dname} {user_iname} {user_interest_text}"
        
        # Load model and vectorize user text
        model = load_xlmr_model()
        user_vector = model.encode(user_text)
        
        print(f"ユーザーテキスト: {user_text}")
        
        # Calculate cosine similarities
        # Using sentence-transformers util for cosine similarity
        similarities = util.pytorch_cos_sim(
            torch.tensor([user_vector]),
            torch.tensor(researcher_vectors)
        )[0].numpy()
        
        # Add similarities to dataframe
        researchers_df['similarity'] = similarities
        
        # Extract research areas
        researchers_df['extracted_areas'] = researchers_df['research_areas'].apply(
            lambda x: parse_research_area(x) if isinstance(x, str) else []
        )
        
        # Check for category matches
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
        
        researchers_df[['subcategory_match', 'major_category_match']] = researchers_df.apply(check_matches, axis=1)
        
        # Adjust similarities based on category matches
        conditions = [
            researchers_df['subcategory_match'],
            researchers_df['major_category_match'] & ~researchers_df['subcategory_match'],
            ~researchers_df['major_category_match'] & ~researchers_df['subcategory_match']
        ]
        
        choices = [
            researchers_df['similarity'] * 1.5,  # Subcategory match
            researchers_df['similarity'] * 0.8,  # Only major category match
            researchers_df['similarity'] * 0.5   # No match
        ]
        
        researchers_df['adjusted_similarity'] = np.select(conditions, choices, default=researchers_df['similarity'])
        
        # Sort by adjusted similarity
        researchers_df = researchers_df.sort_values('adjusted_similarity', ascending=False)
        
        # Return top 10 researchers
        top_researchers = []
        for _, row in researchers_df.head(10).iterrows():
            researcher = row.to_dict()
            researcher['similarity'] = float(researcher['adjusted_similarity'] * 100)  # Convert to percentage
            top_researchers.append(researcher)
            
        return top_researchers
    
    except Exception as e:
        print(f"マッチング処理中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--regenerate-cache', action='store_true', 
                        help='キャッシュを強制的に再生成します')
    args = parser.parse_args()

    if args.regenerate_cache and os.path.exists(VECTOR_CACHE_FILE):
        os.remove(VECTOR_CACHE_FILE)
        print("キャッシュを削除しました。再生成します。")

    if not os.path.exists(DATABASE_RE):
        init_db()
    app.run(debug=True)

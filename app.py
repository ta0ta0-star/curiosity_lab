from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    data = request.form
    name = data.get('name')
    interest = data.get('interest')
    interest_text = data.get('interest_text')
    
    # ここでデータを処理します（例：データベースに保存）
    
    return jsonify({"message": f"登録が完了しました。名前: {name}, 興味: {interest}, 詳細: {interest_text}"}), 200

if __name__ == '__main__':
    app.run(debug=True)
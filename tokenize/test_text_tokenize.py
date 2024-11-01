import pandas as pd
from janome.tokenizer import Tokenizer

# トークン化用のTokenizerの初期化
tokenizer = Tokenizer()

# トークン化関数（名詞を抽出)
def tokenize(text):
    tokens = tokenizer.tokenize(text)
    return [token.base_form for token in tokens if token.part_of_speech.startswith('名詞')]

# 学習用データの読み込み
test_data_path = r"C:\Users\ktg27\intern\task3\bootcamp課題３＿各種データ - 検証用データ.csv"
test_data = pd.read_csv(test_data_path)

# 学習データの概要文をトークン化して新しい列として追加
test_data['tokenized_text'] = test_data['概要文'].apply(tokenize)

# トークン化されたデータをCSVファイルに保存
tokenized_data_path = r"C:\Users\ktg27\intern\task3\tokenized_test_data.csv"
test_data[['tokenized_text', '業界']].to_csv(tokenized_data_path, index=False)

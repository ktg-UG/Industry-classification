import pandas as pd
from janome.tokenizer import Tokenizer

#初期化
tokenizer = Tokenizer()

#トークン化関数（名詞を抽出)
def tokenize(text):
    tokens = tokenizer.tokenize(text)
    noun_tokens = []
    for token in tokens:
        if token.part_of_speech.startswith('名詞'):
            noun_tokens.append(token.base_form)
    return noun_tokens

#読み込み
learning_data_path = r"C:\Users\ktg27\intern\Industry-classification\data\train_data.csv"
learning_data = pd.read_csv(learning_data_path)

#新しい列の追加
learning_data['tokenized_text'] = learning_data['概要文'].apply(tokenize)

#print(learning_data['tokenized_text'].head())

#出力
tokenized_data_path = r"C:\Users\ktg27\intern\Industry-classification\tokenized_data\tokenized_train_data.csv"
learning_data[['tokenized_text', '業界']].to_csv(tokenized_data_path, index=False)

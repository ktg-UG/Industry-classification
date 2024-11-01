#industry-classification
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from janome.tokenizer import Tokenizer
import sys

#ロジスティック回帰
from sklearn.linear_model import LogisticRegression

tokenizer = Tokenizer()

def tokenize(text):
    print(f"トークン化中: {text[:20]}...")
    tokens = tokenizer.tokenize(text)
    print("トークン化完了")
    return [token.base_form for token in tokens if token.part_of_speech.startswith('名詞')]


def main():
    tokenized_learning_data_path = r"C:\Users\ktg27\intern\task3\Industry-classification\tokenized_data\tokenized_learning_data.csv"
    validation_data_path = r"C:\Users\ktg27\intern\task3\Industry-classification\tokenized_data\tokenized_test_data.csv"
    tokenized_learning_data = pd.read_csv(tokenized_learning_data_path)
    validation_data = pd.read_csv(validation_data_path)

    #データの確認
    #print("学習用データのサンプル:")
    #print(learning_data.head())
    #print("\n検証用データのサンプル:")
    #print(validation_data.head())

    #特徴量とラベルに分割
    x_train = tokenized_learning_data['tokenized_text']
    y_train = tokenized_learning_data['業界']

    x_test = validation_data['tokenized_text']
    y_test = validation_data['業界']

    #パイプラインの作成
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000)),
        ('clf', LogisticRegression())
    ])

    #モデルの学習
    print("\nモデルを学習中...")
    pipe.fit(x_train, y_train)

    #検証データでの予測
    print("検証データで予測...")
    y_pred = pipe.predict(x_test)

    #精度の評価
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nモデルの正答率: {accuracy * 100:.2f}%")

    
    #コマンドラインからの入力を受ける
    while True:
        user_input = input("\n企業の概要文を入力してください(終了するには'exit'と入力) : ")
        if user_input.lower() == 'exit':
            print("プログラムを終了します")
            break

        tokenized_input = ' '.join(tokenize(user_input))  # トークン化してスペースで連結
        prediction = pipe.predict([tokenized_input])[0]
        print(f"推定される業界 : {prediction}")

if __name__ == "__main__":
    main()

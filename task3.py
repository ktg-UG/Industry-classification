##task3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
#k近傍法
from sklearn.neighbors import KNeighborsClassifier
#勾配ブースティング
from sklearn.ensemble import GradientBoostingClassifier
#ランダムフォレスト
from sklearn.ensemble import RandomForestClassifier
#ロジスティック回帰 9.6%
from sklearn.linear_model import LogisticRegression
#ナイーブベイズ分類 22.6%
from sklearn.naive_bayes import MultinomialNB
#LinearSVC 13.4%
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

from janome.tokenizer import Tokenizer

import sys

tokenizer = Tokenizer()

def tokenize(text):
    print(f"トークン化中: {text[:20]}...")
    tokens = tokenizer.tokenize(text)
    print("トークン化完了")
    return [token.base_form for token in tokens if token.part_of_speech.startswith('名詞')]


def main():
    #データの読み込み
    try:
        tokenized_learning_data_path = r"C:\Users\ktg27\intern\task3\tokenized_learning_data.csv"
        validation_data_path = r"C:\Users\ktg27\intern\task3\tokenized_test_data.csv"
        tokenized_learning_data = pd.read_csv(tokenized_learning_data_path)
        validation_data = pd.read_csv(validation_data_path)
    except FileNotFoundError:
        print("CSVファイルが見つかりません。ファイルパスを確認してください")
        sys.exit(1)

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
        ('clf', KNeighborsClassifier())
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

    #print("\n詳細な分類レポート:")
    print(classification_report(y_test, y_pred))
    
    if accuracy < 0.7:
        print("正答率が70%未満です。モデルの改善が必要です。")
    else:
        print("正答率が70%です。要件を満たしています。")
    
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

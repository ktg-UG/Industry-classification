import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from janome.tokenizer import Tokenizer
import sys

#ロジスティック回帰 0.938
from sklearn.linear_model import LogisticRegression

def main():
    #データの読み込み
    tokenized_learning_data_path = r"C:\Users\ktg27\intern\task3\tokenized_learning_data.csv"
    validation_data_path = r"C:\Users\ktg27\intern\task3\tokenized_test_data.csv"
    tokenized_learning_data = pd.read_csv(tokenized_learning_data_path)
    validation_data = pd.read_csv(validation_data_path)

    #特徴量とラベルに分割
    X_train = tokenized_learning_data['tokenized_text']
    y_train = tokenized_learning_data['業界']

    X_test = validation_data['tokenized_text']
    y_test = validation_data['業界']

    #パイプラインの作成
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000)),
        ('clf', LogisticRegression())
    ])

    #モデルの学習
    print("\nモデルを学習中...")
    pipe.fit(X_train, y_train)

    print(pipe.score(X_test, y_test))

if __name__ == "__main__":
    main()
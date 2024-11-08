#industry-classification

#サポートベクターマシーン
from sklearn.svm import LinearSVC

#入力した分のトークン化
from janome.tokenizer import Tokenizer
def tokenize(text):
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(text)
    noun_tokens = []
    for token in tokens:
        if token.part_of_speech.startswith('名詞'):
            noun_tokens.append(token.base_form)
    return noun_tokens


def main():
    #データの読み込み
    import pandas as pd
    tokenized_train_data_path = r"tokenized_data\tokenized_train_data.csv"
    tokenized_test_data_path = r"tokenized_data\tokenized_test_data.csv"
    tokenized_train_data = pd.read_csv(tokenized_train_data_path)
    tokenized_test_data = pd.read_csv(tokenized_test_data_path)

    #分割
    x_train = tokenized_train_data['tokenized_text']
    y_train = tokenized_train_data['業界']

    x_test = tokenized_test_data['tokenized_text']
    y_test = tokenized_test_data['業界']

    #テキストのパラメータ化
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    vectorizer.fit(x_train)
    x_train_parameterized = vectorizer.transform(x_train)
    #print(x_train_parameterized)
    
    #業界を数値化
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train_encodered = encoder.transform(y_train)

    #モデルの作成
    svc = LinearSVC()
    svc.fit(x_train_parameterized,y_train_encodered)

    #testデータに対しても行う
    x_test_parameterized = vectorizer.transform(x_test)
    y_test_encodered = encoder.transform(y_test)

    y_predict = svc.predict(x_test_parameterized)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test_encodered, y_predict)
    #print("Accuracy:", accuracy)

    #コマンドラインからの入力を受ける
    while True:
        x_input = input("\n企業の概要文を入力してください(終了するには'exit'と入力) : ")
        if x_input.lower() == 'exit':
            print("プログラムを終了します")
            break

        x_input_tokenized = tokenize(x_input)
        x_input_reshped = [" ".join(x_input_tokenized)]
        x_input_parameterized = vectorizer.transform(x_input_reshped)
        #print(x_input_parameterized)
        y_output_encodered = svc.predict(x_input_parameterized)
        y_output = encoder.inverse_transform(y_output_encodered)
        print(f"推定される業界 : {y_output[0]}")

if __name__ == "__main__":
    main()

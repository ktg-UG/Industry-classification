#k近傍法 0.9
from sklearn.neighbors import KNeighborsClassifier

def main():
    #データの読み込み
    import pandas as pd
    tokenized_learning_data_path = r"C:\Users\ktg27\intern\Industry-classification\tokenized_data\tokenized_learning_data.csv"
    validation_data_path = r"C:\Users\ktg27\intern\Industry-classification\tokenized_data\tokenized_test_data.csv"
    tokenized_learning_data = pd.read_csv(tokenized_learning_data_path)
    validation_data = pd.read_csv(validation_data_path)

    #特徴量とラベルに分割
    x_train = tokenized_learning_data['tokenized_text']
    y_train = tokenized_learning_data['業界']

    x_test = validation_data['tokenized_text']
    y_test = validation_data['業界']

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
    knn = KNeighborsClassifier()
    knn.fit(x_train_parameterized,y_train_encodered)

    #testデータに対しても行う
    x_test_parameterized = vectorizer.transform(x_test)
    y_test_encodered = encoder.transform(y_test)

    y_predict = knn.predict(x_test_parameterized)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test_encodered, y_predict)
    print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()
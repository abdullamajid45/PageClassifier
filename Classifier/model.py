import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import  CountVectorizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.utils import shuffle

class Model():
    def __init__(self):
        data = pd.read_csv("Mydata.csv")
        data1 = data.copy()
        data1=shuffle(data1)
        x = data1["text"].copy()
        y = data1['category'].copy()
#        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
        x=shuffle(x)
        # self.Tfidf_vect = TfidfVectorizer(max_features=50000)
        # self.Tfidf_vect.fit(x.values.astype('U'))
        # Train_X = self.Tfidf_vect.transform(x.values.astype('U'))
        #Test_X = self.Tfidf_vect.transform(x.values.astype('U'))

        self.Count_vect = CountVectorizer()
        Train_X = self.Count_vect.fit_transform(x.values.astype('U'))

        self.Encoder = LabelEncoder()
        Train_Y = self.Encoder.fit_transform(y)
        #Test_Y = self.Encoder.fit_transform(test_y)

        self.classifier_svc = svm.SVC(kernel='linear')
        self.classifier_svc.fit(Train_X, Train_Y)


    def getAccuracy(self):
        return self.accuracy

    def predict(self,data):
        data['text'] = [entry.lower() for entry in data['text']]
        data['text'] = [word_tokenize(entry) for entry in data['text']]

        tag_map = defaultdict(lambda: wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV

        for index, entry in enumerate(data['text']):
            print(data["page"]," ",entry)
            # Declaring Empty List to store the words that follow the rules for this step
            Final_words = []
            # Initializing WordNetLemmatizer()
            word_Lemmatized = WordNetLemmatizer()
            # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
            for word, tag in pos_tag(entry):
                # Below condition is to check for Stop words and consider only alphabets
                if word not in stopwords.words('english') and word.isalpha():
                    word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
                    Final_words.append(word_Final)
            data['text'][index]=Final_words
        data["text"] = [" ".join(entry) for entry in data['text']]
        x = data["text"].copy()
        train_x = self.Count_vect.transform(x)
        y_pred = self.classifier_svc.predict(train_x)
        print(y_pred)
        result=[]
        for i in range(0,len(y_pred)):
            if y_pred[i]==1:
                    result.append(data["page"][i])
        print("Result=========== ",result)
        return result

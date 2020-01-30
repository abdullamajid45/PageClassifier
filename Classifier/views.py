from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter#process_pdf
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from io import StringIO
import re
import pandas as pd
import joblib
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

Count_vect = joblib.load('vectorized_data.sav')
classifier_svc = joblib.load('trained_model.sav')
Encoder = joblib.load('encoded_label.sav')

def pdf_to_text(pdfname):
    # PDFMiner boilerplate

    rsrcmgr = PDFResourceManager()
    codec = 'utf-8'
    laparams = LAParams()
    # Extract text
    fp = open(pdfname, 'rb')
    no=1
    i=0
    text =""
    flag=False
    data = pd.DataFrame([], columns=['page', 'text'])
    for page in PDFPage.get_pages(fp):
        sio = StringIO()
        device = TextConverter(rsrcmgr, sio, codec=codec, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        interpreter.process_page(page)

        text = sio.getvalue()
        text = re.sub('[^A-Za-z0-9 \n]+', '', text)
        sio.close()
        tokenized = text.split()
        if len(tokenized) > 0 and tokenized[-1].isdigit() and len(tokenized[-1]) > 4:
            page_number = int(tokenized[-1])
            if flag==False and (page_number ==1 or page_number ==2):
                i=page_number
                flag=True
            if i>0:
               data = data.append({'page': i, 'text': text}, ignore_index=True)

            if i!=0:
                i+=1
    fp.close()
    # Cleanup
    device.close()
    return data

def index(request):
    return render(request,'index.html')

def output(request):
    file = request.FILES["file"]
    fs = FileSystemStorage()
    name = fs.save(file.name, file)

    try:
        data = pdf_to_text(str(name))

        data['text'] = [entry.lower() for entry in data['text']]
        data['text'] = [word_tokenize(entry) for entry in data['text']]

        tag_map = defaultdict(lambda: wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV

        for index, entry in enumerate(data['text']):
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
            data['text'][index] = Final_words
        data["text"] = [" ".join(entry) for entry in data['text']]
        x = data["text"].copy()
        result=[]
        for item in x:
            result.append(item)

    #     train_x = Count_vect.transform(x)
    #     y_pred = []
    #     if train_x.shape[0] > 0:
    #         y_pred = classifier_svc.predict(train_x)
    #     result = []
    #     for i in range(0, len(y_pred)):
    #         if y_pred[i] == 1:
    #             result.append(data["page"][i])
    #     print("Result=========== ", result)
    #
    except:
        result=[]
    fs.delete(name)

    final = {"data": result}
    return render(request, 'index.html', final)

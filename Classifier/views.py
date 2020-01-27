import time

from django.http import HttpResponse
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter#process_pdf
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from io import StringIO
import re
import pandas as pd

from Classifier.model import Model

classifier=Model()
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
            print(page_number," --- ",i)
            if i!=page_number and i>0:
                print("NO!")
                data = data.append({'page': i, 'text': text}, ignore_index=True)

            else:
                print("YES!")
                data = data.append({'page': page_number, 'text': text}, ignore_index=True)

            i+=1
    fp.close()
    print(data)
    # Cleanup
    device.close()
    return data

def index(request):
    return render(request,'index.html')

def output(request):
    global classifier
    file = request.FILES["file"]
    fs = FileSystemStorage()
    name = fs.save(file.name, file)

    data = pdf_to_text(str(name))
    result = classifier.predict(data)
    fs.delete(name)
    final = {"data": result}
    return render(request, 'index.html', final)
    fs.delete(name)
    final = {"data":[]}
    return render(request,'index.html',final)

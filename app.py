from flask import Flask,render_template,request
from flask_cors import cross_origin
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd=r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

from transformers import AutoTokenizer, BertForQuestionAnswering,pipeline

tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")

model = BertForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")

def qna(cont,que):
  tokenizer.encode(que,truncation=True,padding=True)
  pl=pipeline('question-answering',model=model,tokenizer=tokenizer)
  k=pl({'question':que,'context':cont})
  return k['answer']



app = Flask(__name__) 


@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")

@app.route("/predict",methods =["GET","POST"])
@cross_origin()
def predict():
    if request.method=="POST" :
        context=request.form["context"]
        if(context):
            question=request.form["question"]
            output=qna(context,question)
            return render_template('home.html',output_text=output,context_text=context,question_text=question)
        elif(request.form["myfile"]):
            img=cv2.imread(request.form["myfile"])
            context=pytesseract.image_to_string(img)
            question=request.form["question"]
            output=qna(context,question)
            return render_template('home.html',output_text=output,context_text=context,question_text=question)
        else:
            return render_template("home.html")
    return render_template("home.html")




if __name__=='__main__':
	app.run(debug=True)

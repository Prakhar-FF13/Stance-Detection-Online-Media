# -*- coding: utf-8 -*-

from flask import Flask, render_template,request

import prediction
import pandas as pd
app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('test.html')

@app.route('/',methods=['POST','GET'])
def getValues():
        headl=request.form.get('headline')
        abody=request.form.get('articlebody')
        
        df=pd.DataFrame(columns=['Headline','articleBody'])
        df['Headline']=[headl]
        df['articleBody']=[abody]
        print(df)
        result=prediction.predictOnData(df)
        ans="The Headline and article body "+result[0]+" with each other"
        
        return render_template("test.html",a=ans)
	


if __name__ == "__main__":
    app.run(debug=False)
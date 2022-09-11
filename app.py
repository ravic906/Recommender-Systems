#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template,request
import model

app = Flask(__name__)


@app.route('/')
def mains():
    return render_template('page.html')

@app.route('/submit',methods=["GET","POST"])
def submits():
    first_name = request.form.get("UserID")
    print(first_name)
    return render_template('page.html',output=model.getrecommendationsbyuser(first_name))
if __name__=='__main__':
    app.run()


# In[ ]:





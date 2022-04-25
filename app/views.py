import logging
import numpy as np
import re
import string
from flask import render_template
from flask_wtf import Form
from flask_wtf import FlaskForm
from wtforms import fields
from wtforms.validators import InputRequired
import joblib
from scipy import sparse

from . import app

logger = logging.getLogger('app')


# Flask form 
class PredictForm(FlaskForm):
    category = fields.SelectField('Category', choices=[ ('Sports_and_Outdoors', 'Sports and Outdoors'),	
                                                        ('Automotive', 'Automotive')], validators=[InputRequired()])
    review = fields.TextAreaField('Review:', validators=[InputRequired()])

    submit = fields.SubmitField('Submit')


# Get request handler
@app.get('/')
def home():
    form = PredictForm()
    return render_template('index.html',
        form=form,
        prediction=None,
        prob = None)


# Post request handler
@app.post('/')
def predictions():
    # Get form data
    form = PredictForm()
    target_names = ['Negative', 'Positive']    
    my_proba = None

    # store the submitted values
    submitted_data = form.data
    category = submitted_data['category']
    category_names = ['Sports_and_Outdoors', 'Automotive']

    # Retrieve values from form
    review = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])').sub(r' \1 ', submitted_data['review'])
    for name in category_names:
        if category == name:
            model_loc = 'models/reviews_' + name + "_5.json.gz_model.pkl"
            vec_loc = 'models/reviews_' + name + "_5.json.gz_vector.pkl"
            r_loc = 'models/reviews_' + name + "_5.json.gz_r.npz"
            # unpickle my model
            estimator = joblib.load(model_loc)
            vec = joblib.load(vec_loc)
            r = sparse.load_npz(r_loc)
            break

    review = vec.transform([review])
    my_prediction = estimator.predict(review.multiply(r))
    my_proba = estimator.predict_proba(review.multiply(r))
        
    # Return only the Predicted iris species
    predicted = target_names[int(my_prediction)]
    if my_prediction < 0.5:
        proba = str(round(my_proba[0][0]*100, 2))
    else:
        proba = str(round(my_proba[0][1]*100, 2))
    
    return render_template('index.html', form = form, prediction = predicted, prob = proba)

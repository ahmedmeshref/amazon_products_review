from flask import Flask

app = Flask(__name__)
app.config.from_object("app.config")

from .views import *  

# Handle Not Found Pages Error 
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

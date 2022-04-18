from flask import Flask, render_template
import os

app = Flask(__name__)

plotsPath = 'static/plots'
app.config['UPLOAD_FOLDER'] = plotsPath

@app.route("/")
def index():
    plotsList = os.listdir(plotsPath)
    imagelist = ['plots/' + image for image in plotsList]
    return render_template("index.html", imagelist=imagelist)

app.run()

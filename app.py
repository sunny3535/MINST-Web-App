from flask import Flask , render_template , request
from werkzeug.utils import secure_filename
from PIL import Image
from matplotlib import pyplot
import numpy as np
import tensorflow as tf
#from load_model import model
import os

app = Flask(__name__)
app.debug = True

@app.route("/")
def upload():
    return render_template('index.html')

@app.route('/', methods = ['GET', 'POST'])
def upload_file():
   # Create a directory in a known location to save files to.
   uploads_dir = os.path.join(os.getcwd(), 'static')
   model = tf.keras.models.load_model("mnist_ann_model2.h5")
   if request.method == 'POST':
      f = request.files['file']
      f.save(os.path.join(uploads_dir , secure_filename(f.filename)))


      img = Image.open(f).convert('L')
      img = img.resize((28,28) , Image.ANTIALIAS)
      data = ((np.asarray(img))/255.0)
      pred = model.predict(data.reshape(1,28,28))
      print(pred[0])
      pred = np.argmax(pred, axis=1)

      return render_template('prediction.html' , out = str(pred[0]) , im = f.filename)



if __name__ == '__main__':
   #app.run(debug = True)
   app.run(debug=True, host='0.0.0.0', port=5000)

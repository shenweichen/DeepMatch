# FAQ


## 1. Save or load weights/models
----------------------------------------
To save/load weights,you can write codes just like any other keras models.

```python
model = YoutubeDNN()
model.save_weights('YoutubeDNN_w.h5')
model.load_weights('YoutubeDNN_w.h5')
```

To save/load models,just a little different.

```python
from tensorflow.keras.models import save_model, load_model
model = DeepFM()
save_model(model, 'YoutubeDNN.h5')# save_model, same as before

from deepmatch.layers import custom_objects
model = load_model('YoutubeDNN.h5',custom_objects)# load_model,just add a parameter
```

## 2. Set learning rate and use earlystopping
---------------------------------------------------
You can use any models in DeepMatch like a keras model object.
Here is a example of how to set learning rate and earlystopping:

```python
import deepmatch
from tensorflow.keras.optimizers import Adam, Adagrad
from tensorflow.keras.callbacks import EarlyStopping

model = deepmatch.models.FM(user_feature_columns,item_feature_columns)
model.compile(Adagrad(0.01),'binary_crossentropy',metrics=['binary_crossentropy'])

es = EarlyStopping(monitor='val_binary_crossentropy')
history = model.fit(model_input, data[target].values,batch_size=256, epochs=10, verbose=2, validation_split=0.2,callbacks=[es] )
```

## 3. How to run the demo with GPU ?
Install the TensorFlow build recommended for your CUDA, cuDNN, and platform combination, then install `deepmatch`.

## 4. How to avoid TensorFlow, Keras, h5py, or NumPy compatibility errors?
Install TensorFlow separately before installing DeepMatch. Pick the TensorFlow release according to your Python version, CPU/GPU environment, and platform.

```bash
$ pip install tensorflow
$ pip install deepmatch
```

For Python `>=3.9`, DeepMatch and its dependencies allow modern `h5py` releases with `h5py>=3.7.0`. If TensorFlow reports a NumPy conflict, follow the TensorFlow requirement for the TensorFlow release you installed, for example using `numpy<2` when required by TensorFlow.

Use public `tensorflow.keras` imports in your own code:

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
```

Avoid mixing `tensorflow.python.keras` with `tensorflow.keras`. `tensorflow.python.*` is private TensorFlow API and can break serialization, optimizer loading, or metric loading across TensorFlow versions.

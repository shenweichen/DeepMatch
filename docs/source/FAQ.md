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
from tensorflow.python.keras.models import  save_model,load_model
model = DeepFM()
save_model(model, 'YoutubeDNN.h5')# save_model, same as before

from deepmatch.layers import custom_objects
model = load_model('YoutubeDNN.h5',custom_objects)# load_model,just add a parameter
```

## 2. Set learning rate and use earlystopping
---------------------------------------------------
You can use any models in DeepCTR like a keras model object.
Here is a example of how to set learning rate and earlystopping:

```python
import deepmatch
from tensorflow.python.keras.optimizers import Adam,Adagrad
from tensorflow.python.keras.callbacks import EarlyStopping

model = deepmatch.models.FM(user_feature_columns,item_feature_columns)
model.compile(Adagrad(0.01),'binary_crossentropy',metrics=['binary_crossentropy'])

es = EarlyStopping(monitor='val_binary_crossentropy')
history = model.fit(model_input, data[target].values,batch_size=256, epochs=10, verbose=2, validation_split=0.2,callbacks=[es] )
```

# FAQ

## 1. Set learning rate and use earlystopping
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


# build model 

def base_model():
    inputs = tf.keras.layers.Input(shape=(window_size, 6))
    output = tf.keras.layers.LSTM(units=128, return_sequences=True)(inputs)
    output = tf.keras.layers.Dropout(rate=0.25)(output)
    #output = tf.keras.layers.LSTM(units=128, return_sequences=True)(output)  #try 2 LSTM layers
    output = tf.keras.layers.Dense(units=64, activation='relu')(output)
    output = tf.keras.layers.Dropout(rate=0.25)(output)
    output = tf.keras.layers.Dense(units=12, activation='softmax')(output)
    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    return model

model = base_model()
model.summary()

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#create checkpoints
checkpoint_path = "test_ckpt/cp.ckpt"

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# use fit method to train the model
history = model.fit(
    train_sw,
    epochs=epoch,
    validation_data=val_sw, 
    callbacks=[cp_callback])

# save model and weights
model.save('my_model.h5')
model.save_weights('my_model_weights.h5')

history = history

# plot the training and validation accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.4, 1])
plt.legend(loc='lower right')


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
f = lambda x : x + 1

# calculate test accuracy
def check_testing():
  l_arr = []
  p_arr = []   
  for d,l in test_sw:                            # after unbatch   d(500,6)   l(500,12)
    if len(l_arr) == 0 and len(p_arr)==0:
      pred = model.predict(d)
      p = pred.reshape(-1,12)                    # reduce dimension
      l = tf.reshape(l,(-1,12))    
      p_reformed = f(np.argmax(p, axis=1))       #(40000,)
      l_reformed = f(np.argmax(l, axis=1))       #(40000,)
      l_arr = l_reformed
      p_arr = p_reformed
    else:
      pred = model.predict(d)
      p = pred.reshape(-1,12)                    # reduce dimension
      l = tf.reshape(l,(-1,12))
      p_reformed = f(np.argmax(p, axis=1))      
      l_reformed = f(np.argmax(l, axis=1))

      l_arr = np.hstack((l_arr, l_reformed))
      p_arr = np.hstack((p_arr, p_reformed))     #last (200000,)

  acc_score = accuracy_score(l_arr, p_arr)                        #test accuracy
  recall = recall_score(l_arr, p_arr, average="macro")            #recall
  precision = precision_score(l_arr, p_arr, average="macro")      #precision
  f1 = f1_score(l_arr, p_arr, average="macro")                    #f1 score

  return [acc_score, recall, precision, f1]

print(check_testing())


"""Get sequence after sliding window in array"""

def get_seq_arr(sw):     
  seq_x_arr = []
  seq_y_arr = []
  for x,y in sw:
    x_flatt2 = tf.reshape(x, [-1,6]) 
    y_flatt2 = tf.reshape(y, [-1,12]) 
    seq_x_arr.append(x_flatt2)
    seq_y_arr.append(y_flatt2)
  return seq_x_arr, seq_y_arr

seq_x_arr,seq_y_arr = get_seq_arr(test_sw)
print(len(seq_x_arr))   # len = 8 
print(len(seq_y_arr))   # len = 8 

# print(tf.shape(seq_x_arr[0]))    #  [40000     6]




"""Get the whole sequence of test dataset"""

def build_seq(x_arr,y_arr):
  if len(x_arr) == len(y_arr):
    for i in range(0, len(x_arr)):
      if i == 0:
        seq_x_concat = x_arr[i]
        seq_y_concat = y_arr[i]
      else:
        seq_x_next = x_arr[i]
        seq_y_next = y_arr[i]
        seq_x_concat = tf.concat([seq_x_concat,seq_x_next],0)
        seq_y_concat = tf.concat([seq_y_concat,seq_y_next],0)
  else:
    print('Error. Check function get_seq_arr().')
  
  return seq_x_concat, seq_y_concat

seq_x,seq_y = build_seq(seq_x_arr,seq_y_arr)
# print(tf.shape(seq_x))     # [320000      6]
# print(tf.shape(seq_y))     # [320000      12]

# take out the data of x,y,z acceleration seperately
acc_X = seq_x[:,0]
acc_Y = seq_x[:,1]
acc_Z = seq_x[:,2]

# plot x,y,z acceleration in one figure
fig_acc = plt.figure(figsize=(50,10))
a = fig_acc.add_subplot(3,1,1)
a.set_title('Accelorometer')
plt.plot(acc_X,linewidth=0.3,label='acc_X')
plt.plot(acc_Y,linewidth=0.3,label='acc_Y')
plt.plot(acc_Z,linewidth=0.3,label='acc_Z')
plt.ylabel('Acceleration [g]')
plt.xlabel('Time interval') 
plt.legend()
plt.show()

# take out the data of x,y,z angular velocity seperately
gyro_X = seq_x[:,3]
gyro_Y = seq_x[:,4]
gyro_Z = seq_x[:,5]

#plot x,y,z angular velocity in one figure
fig_gyro = plt.figure(figsize=(50,10))
b = fig_gyro.add_subplot(3,1,2)
b.set_title('Gyroscope')
plt.plot(gyro_X,linewidth=0.3,label='gyro_X')
plt.plot(gyro_Y,linewidth=0.3,label='gyro_Y')
plt.plot(gyro_Z,linewidth=0.3,label='gyro_Z')
plt.ylabel('Angular Velocity [rad/sec]')
plt.xlabel('Time interval') 
plt.legend()
plt.show()

# get predicted labels of test dataset
f = lambda x: x+1
pred = model.predict(test_sw)
pred = tf.reshape(pred, [-1,12])         # (320000, 12)
pred_label = np.argmax(pred, axis = 1)   # change the prediction to class number
pred_label = f(np.array(pred_label).reshape(pred_label.size,)) # convert labels to one dimension

# get true label within one window
label_true = np.argmax(seq_y, axis=1)
label_true = f(np.array(label_true).reshape(label_true.size,))

# plot true label and predicted label in one figure
fig_label = plt.figure(figsize=(50,10))
c = fig_label.add_subplot(3,1,3)
c.set_title('Class')
plt.plot(label_true,linewidth=0.3,label='label_true')
plt.plot(pred_label,linewidth=0.3,label='label_pred')
plt.ylabel('Class')
plt.xlabel('Time interval')
plt.legend()
plt.show()




"""Get samples within one window length"""

for x,y in test_sw.unbatch().take(1):
  seq_x = x  
  seq_y = y

for x,y in test_sw.take(1):
  seq_x_batch = x  
  seq_y_batch = y

# take out the data of x,y,z acceleration within one window seperately
acc_X = seq_x[:,0]
acc_Y = seq_x[:,1]
acc_Z = seq_x[:,2]

# plot x,y,z acceleration within one window in one figure
fig_acc = plt.figure(figsize=(20,10))
a = fig_acc.add_subplot(3,1,1)
a.set_title('Accelorometer')
plt.plot(acc_X,linewidth=0.5,label='acc_X')
plt.plot(acc_Y,linewidth=0.5,label='acc_Y')
plt.plot(acc_Z,linewidth=0.5,label='acc_Z')
plt.ylabel('Acceleration [g]')
plt.xlabel('Time interval') 
plt.legend()
plt.show()


# take out the data of x,y,z angular velocity within one window seperately
gyro_X = seq_x[:,3]
gyro_Y = seq_x[:,4]
gyro_Z = seq_x[:,5]    #(175059, )

# plot x,y,z angular velocity within one window in one figure
fig_gyro = plt.figure(figsize=(20,10))
b = fig_gyro.add_subplot(3,1,2)
b.set_title('Gyroscope')
plt.plot(gyro_X,linewidth=0.5,label='gyro_X')
plt.plot(gyro_Y,linewidth=0.5,label='gyro_Y')
plt.plot(gyro_Z,linewidth=0.5,label='gyro_Z')
plt.ylabel('Angular Velocity [rad/sec]')
plt.xlabel('Time interval')
plt.legend() 
plt.show()

# get predicted labels of one window
f = lambda x: x+1
test_sample = test_sw.take(1)                                    # take one sample of unbatched test dataset
pred = model.predict(test_sample)
pred_label = np.argmax(pred, axis = 2)                           # predicted label of this sample
pred_label = f(np.array(pred_label).reshape(pred_label.size,))   # convert labels to one dimension
print(np.shape(pred_label))

# get true labels of one window
label_true = np.argmax(seq_y_batch, axis=2)
label_true = np.array(pred_label).reshape(pred_label.size,)

# # plot true label and predicted label of one window in one figure
fig_label = plt.figure(figsize=(20,10))
c = fig_label.add_subplot(3,1,2)
c.set_title('Label')
plt.plot(label_true,linewidth=0.5,label='label_true')
plt.plot(pred_label,linewidth=0.5,label='label_pred')
plt.ylabel('Label')
plt.xlabel('Time interval') 
plt.legend() 
plt.show()


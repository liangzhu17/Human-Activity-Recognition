
# define a function to get true labels and predicted labels
def get_l_p():
  l_arr = []
  p_arr = []
    
  for d,l in test_sw: 
    if len(l_arr) == 0 and len(p_arr)==0:
      pred = model.predict(d)
      p = pred.reshape(-1,12)                 # reduce dimension
      l = tf.reshape(l,(-1,12))    
      p_reformed = f(np.argmax(p, axis=1))    #(40000,)
      l_reformed = f(np.argmax(l, axis=1))    #(40000,)
      l_arr = l_reformed
      p_arr = p_reformed
    else:
      pred = model.predict(d)
      p = pred.reshape(-1,12)                 # reduce dimension
      l = tf.reshape(l,(-1,12))
      p_reformed = f(np.argmax(p, axis=1))      
      l_reformed = f(np.argmax(l, axis=1))

      l_arr = np.hstack((l_arr, l_reformed))
      p_arr = np.hstack((p_arr, p_reformed))   #last (200000,)
  return l_arr, p_arr


# Confusion matrix
def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # the probabilities in confusion matrix
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=5, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()


classes = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING', 'STAND_TO_SIT',
           'SIT_TO_STAND', 'SIT_TO_LIE', 'LIE_TO_SIT', 'STAND_TO_SIT', 'LIE_TO_STAND']

y_true, y_pred = get_l_p()
#print(np.array(y_true).shape)
#print(np.array(y_pred).shape)


# plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm, 'confusion_matrix.png', title='confusion matrix')

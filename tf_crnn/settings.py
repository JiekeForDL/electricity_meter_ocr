train_path = 'crnn_images'
sequence_length = 29
image_height = 36
image_width = image_height * 15
channels = 3

batch_size = 100
epochs = 500
learning_rate = 0.001
lr_decay = 1e-6
labels = list('0123456789DFNOSY.')
num_output = len(labels) + 1

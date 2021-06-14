import tensorflow as tf
import os
import json
import time
from pprint import pprint

print(tf.test.is_gpu_available())

# CONFIG PART

BATCH_SIZE = 32
LEARNING_RATE = 2.5e-4
EMBEDDING_SIZE = 64
LSTM_UNITS = 256

TRAIN_DATA_DIR = "D:\\FigureQA\\train1"
# TRAIN_DATA_DIR = "D:\\FigureQA\\sample_train1"
VALID_DATA_DIR = "D:\\FigureQA\\validation2"

# preprocess data
# json data is composed like this:
# annotations['qa_pairs'] == list
# annotations['qa_pairs'][0] =
#         {'answer': 0,
#          'color1_id': 28,
#          'color1_name': 'Dark Turquoise',
#          'color1_rgb': [0, 206, 209],
#          'color2_id': -1,
#          'color2_name': '--None--',
#          'color2_rgb': [-1, -1, -1],
#          'image_index': 0,
#          'question_id': 0,
#          'question_string': 'Is Dark Turquoise the minimum?'}

def get_dataset(dir, istraining=True):
    with open(os.path.join(dir, 'qa_pairs.json')) as file:
        annotations = json.load(file)
        zipped = zip(*[(q['question_string'], q['answer'], q['image_index'])
                          for q in annotations['qa_pairs']])
        # convert each element into list
        questions, answers, images = [list(x) for x in zipped]

        # successfully extract questions from json file

        # from_tensor_slices(( (1,2), (2,3)) will not work
        # element inside `()` must be list not tuple
        ds = tf.data.Dataset.from_tensor_slices((questions, answers)).batch(BATCH_SIZE)
        return ds

train_ds = get_dataset(TRAIN_DATA_DIR)
val_ds = get_dataset(VALID_DATA_DIR)

train_questions = train_ds.map(lambda x, y: x)
max_q_length = int(max(list(iter(train_questions.map(len)))))
print('max question length: ', max_q_length)

AUTOTUNE = tf.data.AUTOTUNE

batch_size = BATCH_SIZE
seed = 42

# according to the baseline model, dict_size is flexible calculated runtime
# max_features = 10000
max_features = None
sequence_length = max_q_length


# TODO: parameter should be restricted or not
vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=max_features,
    output_sequence_length=sequence_length
)

vectorize_layer.adapt(train_questions)
del train_questions

dict_size = len(vectorize_layer.get_vocabulary())
print(vectorize_layer.get_vocabulary())

# finish vocab creation, tokenize dataset

def vectorize_question(question, answer):
    question = tf.expand_dims(question, -1)
    answer = tf.expand_dims(answer, -1)
    return vectorize_layer(question), answer

train_ds = train_ds.map(vectorize_question)
val_ds = val_ds.map(vectorize_question)

# TODO: does baseline use SGD or not?
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

embedding_dim = EMBEDDING_SIZE

print(dict_size, max_features)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(dict_size + 1, embedding_dim),
    tf.keras.layers.LSTM(256),
    tf.keras.layers.Dense(512),
    tf.keras.layers.Dense(512),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])

model.summary()

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer='Adam',
    # optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=tf.keras.metrics.BinaryAccuracy()
)

start_tm = time.time()
epochs = 10
history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = epochs,
)

end_tm = time.time()
print('training time: ', end_tm - start_tm)
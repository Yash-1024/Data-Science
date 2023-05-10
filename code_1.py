import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, f1_score

# Set up data generators for training and validation sets
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
batch_size = 32
train_generator = datagen.flow_from_directory(
    r'E:\ECE\6th_sem\mini project\dataset',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)
val_generator = datagen.flow_from_directory(
    r'E:\ECE\6th_sem\mini project\dataset',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Make predictions on the test set and generate confusion matrix
test_generator = datagen.flow_from_directory(
    r'E:\ECE\6th_sem\mini project\dataset',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

test_predictions = model.predict(test_generator)
test_labels = test_generator.classes
test_cm = confusion_matrix(test_labels, test_predictions.round())
print("Confusion Matrix:\n", test_cm)
print("f1_score :",f1_score(test_labels, test_predictions.round()))


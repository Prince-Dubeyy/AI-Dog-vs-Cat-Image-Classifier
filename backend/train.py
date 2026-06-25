import tensorflow as tf
import tensorflow_datasets as tfds
import os

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

def load_data():
    print("Loading oxford_iiit_pet dataset...")
    # Proper train/validation split using existing dataset capabilities
    train_ds, val_ds = tfds.load('oxford_iiit_pet', split=['train', 'test'], as_supervised=False)
    
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    
    def preprocess(features):
        image = features['image']
        label = features['species'] # 0 for Cat, 1 for Dog
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.float32)
        return preprocess_input(image), label

    train_dataset = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    validation_dataset = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, validation_dataset

def build_model():
    print("Building model...")
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    
    # Freeze the base_model
    base_model.trainable = False
    
    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ])

    # Create the classification head
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=IMG_SHAPE),
        data_augmentation,
        base_model,
        global_average_layer,
        tf.keras.layers.Dropout(0.2),
        prediction_layer
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    
    return model, base_model

def main():
    train_dataset, validation_dataset = load_data()
    model, base_model = build_model()
    
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "dog_cat_classifier.keras"))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=True
    )
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-6
    )

    print("Starting initial training...")
    history = model.fit(train_dataset,
                        epochs=10,
                        validation_data=validation_dataset,
                        callbacks=[early_stopping, model_checkpoint, reduce_lr])
    
    print("Starting fine-tuning...")
    # Unfreeze the base model
    base_model.trainable = True
    
    # Freeze the bottom 100 layers and fine-tune the rest
    for layer in base_model.layers[:100]:
        layer.trainable = False
        
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
                  
    history_fine = model.fit(train_dataset,
                             epochs=10,
                             validation_data=validation_dataset,
                             callbacks=[early_stopping, model_checkpoint, reduce_lr])
    
    print(f"Model successfully saved to {save_path}")

if __name__ == '__main__':
    main()

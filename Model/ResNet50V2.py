import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

DATASET_DIR = "garbage_classification"

# Hyperparameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 12
SEED = 42

# Training parameters
PHASE1_EPOCHS = 15
PHASE1_LR = 1e-3
PHASE2_EPOCHS = 15
PHASE2_LR = 1e-5
UNFREEZE_LAYERS = 50  # Number of layers to unfreeze in phase 2

# Output paths
MODEL_SAVE_PATH = "resnet50v2_garbage_classifier.keras"
SAVEDMODEL_PATH = "resnet50v2_garbage_classifier_savedmodel"
CHECKPOINT_PATH = "ai_model_checkpoint.keras"


def create_datasets(dataset_dir, image_size, batch_size, seed=42):
    """
    Create train, validation, and test datasets from directory.
    Split: 80% train, 10% validation, 10% test
    """
    print("=" * 60)
    print("Loading and splitting dataset...")
    print("=" * 60)
    
    # Load training set (80%)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        label_mode='int'
    )
    
    # Load remaining 20% for validation + test split
    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        label_mode='int'
    )
    
    # Get class names
    class_names = train_ds.class_names
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Split the 20% into 10% val and 10% test
    val_batches = int(len(val_test_ds) * 0.5)
    val_ds = val_test_ds.take(val_batches)
    test_ds = val_test_ds.skip(val_batches)
    
    print(f"Training batches: {len(train_ds)}")
    print(f"Validation batches: {len(val_ds)}")
    print(f"Test batches: {len(test_ds)}")
    
    return train_ds, val_ds, test_ds, class_names


def create_data_augmentation():
    """
    Create data augmentation layer for training.
    Includes horizontal flip, rotation, and zoom.
    """
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.1),
        layers.RandomBrightness(0.1),
    ], name='data_augmentation')
    
    return data_augmentation


def create_preprocessing_layer():
    """
    Create preprocessing layer using ResNet50V2 preprocessing function.
    This normalizes pixel values appropriately for ResNet50V2.
    """
    def preprocess(images, labels):
        images = preprocess_input(images)
        return images, labels
    
    return preprocess


def build_model(num_classes, data_augmentation):
    """
    Build the ResNet50V2 model with custom classification head.
    
    Architecture:
    - Data Augmentation (only during training)
    - ResNet50V2 Base (pretrained, frozen initially)
    - GlobalAveragePooling2D
    - Dropout (0.3)
    - Dense output layer (num_classes, softmax)
    """
    print("\n" + "=" * 60)
    print("Building ResNet50V2 Model...")
    print("=" * 60)
    
    # Load pretrained ResNet50V2 without top classification layer
    base_model = ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )
    
    # Freeze the base model for initial training
    base_model.trainable = False
    print(f"Base model layers: {len(base_model.layers)}")
    print("Base model frozen for Phase 1 training")
    
    # Build the complete model
    inputs = layers.Input(shape=(224, 224, 3))
    
    # Data augmentation (only active during training)
    x = data_augmentation(inputs)
    
    # Preprocessing for ResNet50V2
    x = preprocess_input(x)
    
    # Pass through base model
    x = base_model(x, training=False)
    
    # Classification head
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = models.Model(inputs, outputs, name='ResNet50V2_GarbageClassifier')
    
    print("\nModel Summary:")
    model.summary()
    
    return model, base_model


def get_callbacks(checkpoint_path):
    """
    Create callbacks for training:
    - ModelCheckpoint: Save best model based on val_accuracy
    - EarlyStopping: Stop training if no improvement for 5 epochs
    """
    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            mode='max',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]
    return callbacks


def train_phase1(model, train_ds, val_ds, epochs, learning_rate, checkpoint_path):
    """
    Phase 1: Fine-tune only the classification head.
    Base model is frozen, training only the top layers.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Training Classification Head (Base Model Frozen)")
    print("=" * 60)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = get_callbacks(checkpoint_path)
    
    # Train
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\nPhase 1 Complete!")
    print(f"Best validation accuracy: {max(history1.history['val_accuracy']):.4f}")
    
    return history1


def train_phase2(model, base_model, train_ds, val_ds, epochs, learning_rate, 
                 unfreeze_layers, checkpoint_path):
    """
    Phase 2: Fine-tune deeper layers of ResNet50V2.
    Unfreeze the last N layers of the base model for fine-tuning.
    """
    print("\n" + "=" * 60)
    print(f"PHASE 2: Fine-Tuning (Unfreezing last {unfreeze_layers} layers)")
    print("=" * 60)
    
    # Unfreeze the base model
    base_model.trainable = True
    
    # Freeze all layers except the last unfreeze_layers
    total_layers = len(base_model.layers)
    freeze_until = total_layers - unfreeze_layers
    
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    
    trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
    print(f"Total base model layers: {total_layers}")
    print(f"Trainable layers after unfreezing: {trainable_count}")
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = get_callbacks(checkpoint_path)
    
    # Continue training
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\nPhase 2 Complete!")
    print(f"Best validation accuracy: {max(history2.history['val_accuracy']):.4f}")
    
    return history2


def evaluate_model(model, test_ds, class_names):
    """
    Evaluate the model on the test set.
    Compute accuracy, loss, and detailed classification report.
    """
    print("\n" + "=" * 60)
    print("Evaluating Model on Test Set")
    print("=" * 60)
    
    # Evaluate overall performance
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Get predictions for classification report
    print("\nGenerating Classification Report...")
    
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        y_true.extend(labels.numpy())
        y_pred.extend(predicted_classes)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Classification Report
    print("\n" + "-" * 60)
    print("Classification Report:")
    print("-" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion Matrix
    print("\n" + "-" * 60)
    print("Confusion Matrix:")
    print("-" * 60)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    return test_loss, test_accuracy


def save_model(model, h5_path, savedmodel_path):
    """
    Save the trained model in both .keras and SavedModel formats.
    """
    print("\n" + "=" * 60)
    print("Saving Model")
    print("=" * 60)
    
    # Save as .h5
    model.save(h5_path)
    print(f"Model saved as HDF5: {h5_path}")
    
    # Save as SavedModel format (for TFLite/TFServing)
    model.export(savedmodel_path)
    print(f"Model exported as SavedModel: {savedmodel_path}")


def main():
    """
    Main function to orchestrate the training pipeline.
    """
    print("\n" + "=" * 60)
    print("Garbage Classification Training with ResNet50V2")
    print("=" * 60)
    
    # Check if dataset directory exists
    if not os.path.exists(DATASET_DIR):
        print(f"\nError: Dataset directory not found: {DATASET_DIR}")
        print("Please set the DATASET_DIR variable to your dataset path.")
        return
    
    # Set random seeds for reproducibility
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\nGPU detected: {gpus}")
        # Enable memory growth to prevent OOM
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("\nNo GPU detected, using CPU")
    
    # Step 1: Create datasets
    train_ds, val_ds, test_ds, class_names = create_datasets(
        DATASET_DIR, IMAGE_SIZE, BATCH_SIZE, SEED
    )
    
    # Verify number of classes
    if len(class_names) != NUM_CLASSES:
        print(f"\nWarning: Expected {NUM_CLASSES} classes, found {len(class_names)}")
    
    # Optimize dataset performance with prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
    
    # Step 2: Create data augmentation layer
    data_augmentation = create_data_augmentation()
    
    # Step 3: Build model
    model, base_model = build_model(len(class_names), data_augmentation)
    
    # Step 4: Phase 1 Training (frozen base model)
    history1 = train_phase1(
        model, train_ds, val_ds,
        epochs=PHASE1_EPOCHS,
        learning_rate=PHASE1_LR,
        checkpoint_path=CHECKPOINT_PATH
    )
    
    # Step 5: Phase 2 Training (fine-tune deeper layers)
    history2 = train_phase2(
        model, base_model, train_ds, val_ds,
        epochs=PHASE2_EPOCHS,
        learning_rate=PHASE2_LR,
        unfreeze_layers=UNFREEZE_LAYERS,
        checkpoint_path=CHECKPOINT_PATH
    )
    
    # Step 6: Evaluate on test set
    test_loss, test_accuracy = evaluate_model(model, test_ds, class_names)
    
    # Step 7: Save final model
    save_model(model, MODEL_SAVE_PATH, SAVEDMODEL_PATH)
    
    # Final summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"SavedModel saved to: {SAVEDMODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()

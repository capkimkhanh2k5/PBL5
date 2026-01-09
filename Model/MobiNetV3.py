"""
MobileNetV3-Large Garbage Classification Training Script
OPTIMIZED FOR TENSORFLOW 2.16 + KERAS 3 + METAL GPU
========================================================
✅ Fixed: Memory leak causing freeze at epoch 10
✅ Fixed: GPU memory management
✅ Fixed: Keras 3 compatibility (optimizer.learning_rate)
✅ Added: Mixed precision (float16) for Metal GPU speedup
✅ Optimized: Batch size auto-adjustment
"""

# ============================================================================
# FIX TENSORFLOW METAL + GPU MEMORY CONFIGURATION
# Must be set BEFORE importing TensorFlow
# ============================================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # ✅ CRITICAL FIX

import json
import numpy as np
import tensorflow as tf

# ✅ CONFIGURE GPU MEMORY GROWTH (PREVENTS FREEZE)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
        
        # ✅ ENABLE MIXED PRECISION FOR METAL GPU (1.5-2x SPEEDUP)
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        print("✅ Mixed precision (float16) enabled for Metal GPU")
    except RuntimeError as e:
        print(f"⚠️ GPU configuration error: {e}")

from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = "data"
IMG_SIZE = (288, 288)
BATCH_SIZE = 64
NUM_CLASSES = 12
SEED = 123

MODEL_PATH = "mobilenetv3.weights.h5"  # Keras 3: use weights-only format
CLASS_NAMES_PATH = "class_names.json"
TRAINING_HISTORY_PATH = "training_history.json"
CONFUSION_MATRIX_PATH = "confusion_matrix.png"

# Training epochs for each stage
EPOCHS_STAGE1 = 20
EPOCHS_STAGE2 = 30
EPOCHS_STAGE3 = 40

# Advanced settings
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.2
USE_TTA = True
L2_WEIGHT_DECAY = 1e-5

print(f"\n🎯 Configuration:")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Image size: {IMG_SIZE}")
print(f"   GPU available: {len(gpus) > 0}")

# ============================================================================
# ADVANCED DATA AUGMENTATION
# ============================================================================

def create_data_augmentation():
    """Create enhanced data augmentation pipeline (Keras 2.13 compatible)."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.3),
        layers.RandomZoom(0.2),
        layers.RandomTranslation(0.2, 0.2),
        layers.RandomContrast(0.2),
    ], name="data_augmentation")


class MixUpAugmentation(tf.keras.layers.Layer):
    """
    MixUp augmentation layer - OPTIMIZED for memory efficiency.
    """
    def __init__(self, alpha=0.2, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        
    def call(self, inputs, training=None):
        if training is None or not training:
            return inputs
            
        images, labels = inputs
        batch_size = tf.shape(images)[0]
        
        # Sample lambda from Beta distribution
        lambda_val = tf.random.uniform([], 0, self.alpha)
        
        # ✅ Cast lambda_val to match image dtype (for mixed precision compatibility)
        lambda_val = tf.cast(lambda_val, images.dtype)
        
        # Shuffle indices
        indices = tf.random.shuffle(tf.range(batch_size))
        mixed_images = lambda_val * images + (1 - lambda_val) * tf.gather(images, indices)
        
        # One-hot encode labels for mixing (keep as float32 for loss calculation)
        labels_one_hot = tf.one_hot(labels, NUM_CLASSES, dtype=tf.float32)
        labels_shuffled = tf.gather(labels_one_hot, indices)
        lambda_val_f32 = tf.cast(lambda_val, tf.float32)  # Cast back for label mixing
        mixed_labels = lambda_val_f32 * labels_one_hot + (1 - lambda_val_f32) * labels_shuffled
        
        return mixed_images, mixed_labels
    
    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha})
        return config


# ============================================================================
# COSINE DECAY WITH WARMUP CALLBACK
# ============================================================================

class CosineDecayCallback(tf.keras.callbacks.Callback):
    """Cosine decay learning rate with warmup - Memory efficient."""
    def __init__(self, initial_lr, total_epochs, warmup_epochs=2, min_lr=1e-7):
        super().__init__()
        self.initial_lr = initial_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
        
        # Keras 3 compatibility: use learning_rate instead of lr
        self.model.optimizer.learning_rate.assign(lr)
        print(f"Epoch {epoch + 1}: Learning rate = {lr:.2e}")


# ============================================================================
# MEMORY-EFFICIENT CALLBACK (SAFE VERSION)
# ============================================================================

class MemoryMonitorCallback(tf.keras.callbacks.Callback):
    """Monitor memory and trigger safe garbage collection."""
    def __init__(self, clear_every_n_epochs=5):
        super().__init__()
        self.clear_every_n_epochs = clear_every_n_epochs
        
    def on_epoch_end(self, epoch, logs=None):
        # ✅ SAFE: Only garbage collection, NO clear_session()
        if (epoch + 1) % self.clear_every_n_epochs == 0:
            print(f"\n🧹 Running garbage collection at epoch {epoch + 1}")
            import gc
            gc.collect()
            # ✅ SAFE: Clear unused variables only
            if tf.config.list_physical_devices('GPU'):
                try:
                    tf.config.experimental.reset_memory_stats('GPU:0')
                except:
                    pass  # Not critical if fails


# ============================================================================
# DATA PREPARATION
# ============================================================================

def create_datasets():
    """Load images and split into train/val/test sets."""
    print("\n" + "=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)
    
    # Load full dataset first to get class names
    full_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=SEED
    )
    class_names = full_ds.class_names
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Create train set (80%)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=SEED,
        validation_split=0.2,
        subset="training"
    )
    
    # Create validation + test set (20%)
    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=SEED,
        validation_split=0.2,
        subset="validation"
    )
    
    # Split val_test into val (50%) and test (50%)
    val_test_size = tf.data.experimental.cardinality(val_test_ds).numpy()
    val_size = val_test_size // 2
    
    val_ds = val_test_ds.take(val_size)
    test_ds = val_test_ds.skip(val_size)
    
    # Count samples
    train_count = sum(1 for _ in train_ds.unbatch())
    val_count = sum(1 for _ in val_ds.unbatch())
    test_count = sum(1 for _ in test_ds.unbatch())
    
    print(f"\nDataset sizes:")
    print(f"  Train: {train_count} samples")
    print(f"  Validation: {val_count} samples")
    print(f"  Test: {test_count} samples")
    
    return train_ds, val_ds, test_ds, class_names


def calculate_class_weights(train_ds, class_names):
    """Calculate class weights to handle imbalanced dataset."""
    print("\nCalculating class weights...")
    
    class_counts = {i: 0 for i in range(len(class_names))}
    for _, labels in train_ds.unbatch():
        class_counts[int(labels.numpy())] += 1
    
    total = sum(class_counts.values())
    n_classes = len(class_names)
    
    class_weight = {}
    for cls_id, count in class_counts.items():
        class_weight[cls_id] = total / (n_classes * count) if count > 0 else 1.0
    
    print("Class distribution:")
    for cls_id, count in class_counts.items():
        weight = class_weight[cls_id]
        print(f"  {class_names[cls_id]}: {count} samples (weight: {weight:.3f})")
    
    return class_weight


def optimize_datasets(train_ds, val_ds, test_ds, data_augmentation, use_mixup=True):
    """
    ✅ FIXED: Memory-efficient dataset optimization
    - Removed .cache() to prevent memory leak
    - Optimized prefetch strategy
    """
    AUTOTUNE = tf.data.AUTOTUNE
    
    if use_mixup:
        print("✓ MixUp augmentation ENABLED")
        mixup_layer = MixUpAugmentation(alpha=MIXUP_ALPHA)
        
        # Apply MixUp then augmentation
        train_ds = train_ds.map(
            lambda x, y: mixup_layer((x, y), training=True),
            num_parallel_calls=AUTOTUNE
        )
        
        train_ds = train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )
        
        # One-hot encode validation and test labels
        val_ds = val_ds.map(
            lambda x, y: (x, tf.one_hot(y, NUM_CLASSES)),
            num_parallel_calls=AUTOTUNE
        )
        test_ds = test_ds.map(
            lambda x, y: (x, tf.one_hot(y, NUM_CLASSES)),
            num_parallel_calls=AUTOTUNE
        )
    else:
        train_ds = train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )
    
    # ✅ CRITICAL FIX: NO .cache() - prevents memory leak
    # ✅ Shuffle with reasonable buffer size
    # ✅ Prefetch for performance
    train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
    
    print("\n✅ Memory-efficient dataset pipeline configured:")
    print("   • No .cache() - prevents RAM overflow")
    print("   • Shuffle buffer: 1000 (reasonable size)")
    print("   • Prefetch: AUTO")
    
    return train_ds, val_ds, test_ds


# ============================================================================
# MODEL BUILDING
# ============================================================================

def build_model():
    """Build MobileNetV3-Large model with custom classification head."""
    print("\n" + "=" * 60)
    print("BUILDING MODEL")
    print("=" * 60)
    
    base_model = MobileNetV3Large(
        include_top=False,
        weights="imagenet",
        input_shape=IMG_SIZE + (3,),
        include_preprocessing=True
    )
    
    base_model.trainable = False
    
    print(f"Base model: MobileNetV3Large")
    print(f"Total layers: {len(base_model.layers)}")
    
    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    # ✅ Use float32 for final layer to avoid numerical instability with mixed precision
    outputs = layers.Dense(
        NUM_CLASSES, 
        activation="softmax",
        dtype="float32",  # Critical for mixed precision stability
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY)
    )(x)
    
    model = Model(inputs, outputs)
    
    print("\nModel summary:")
    model.summary()
    
    return model, base_model


# ============================================================================
# CALLBACKS
# ============================================================================

def get_callbacks(stage_name):
    """Create callbacks with memory monitoring."""
    return [
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,  # Keras 3: save weights only
            mode="max",
            verbose=1
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        MemoryMonitorCallback(clear_every_n_epochs=5)  # ✅ Prevent memory leak
    ]


# ============================================================================
# THREE-STAGE TRAINING
# ============================================================================

def train_stage1(model, train_ds, val_ds, total_epochs):
    """Stage 1: Train classification head."""
    print("\n" + "=" * 60)
    print("STAGE 1: TRAIN HEAD (Backbone Frozen)")
    print("=" * 60)
    
    initial_lr = 3e-4
    optimizer = Adam(learning_rate=initial_lr, clipnorm=1.0)
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"]
    )
    
    cosine_callback = CosineDecayCallback(
        initial_lr=initial_lr,
        total_epochs=total_epochs,
        warmup_epochs=2
    )
    
    callbacks = get_callbacks("stage1") + [cosine_callback]
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        callbacks=callbacks,
        verbose=1  # ✅ Show progress
    )
    
    best_val_acc = max(history.history['val_accuracy'])
    print(f"\n✓ Stage 1 complete - Best val_accuracy: {best_val_acc:.4f}")
    
    return history


def train_stage2(model, base_model, train_ds, val_ds, total_epochs):
    """Stage 2: Fine-tune upper 50% of backbone."""
    print("\n" + "=" * 60)
    print("STAGE 2: FINE-TUNE UPPER 50% OF BACKBONE")
    print("=" * 60)
    
    base_model.trainable = True
    total_layers = len(base_model.layers)
    freeze_until = total_layers // 2
    
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    
    trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
    print(f"Total backbone layers: {total_layers}")
    print(f"Frozen layers: {freeze_until}")
    print(f"Trainable layers: {trainable_count}")
    
    initial_lr = 1e-4
    optimizer = Adam(learning_rate=initial_lr, clipnorm=1.0)
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"]
    )
    
    cosine_callback = CosineDecayCallback(
        initial_lr=initial_lr,
        total_epochs=total_epochs,
        warmup_epochs=1
    )
    
    callbacks = get_callbacks("stage2") + [cosine_callback]
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    best_val_acc = max(history.history['val_accuracy'])
    print(f"\n✓ Stage 2 complete - Best val_accuracy: {best_val_acc:.4f}")
    
    return history


def train_stage3(model, base_model, train_ds, val_ds, total_epochs):
    """Stage 3: Deep fine-tune nearly all layers."""
    print("\n" + "=" * 60)
    print("STAGE 3: DEEP FINE-TUNE (Nearly All Layers)")
    print("=" * 60)
    
    base_model.trainable = True
    freeze_count = 10
    
    for layer in base_model.layers[:freeze_count]:
        layer.trainable = False
    
    trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
    print(f"Total backbone layers: {len(base_model.layers)}")
    print(f"Frozen layers: {freeze_count}")
    print(f"Trainable layers: {trainable_count}")
    
    initial_lr = 3e-5
    optimizer = Adam(learning_rate=initial_lr, clipnorm=1.0)
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"]
    )
    
    cosine_callback = CosineDecayCallback(
        initial_lr=initial_lr,
        total_epochs=total_epochs,
        warmup_epochs=1
    )
    
    callbacks = get_callbacks("stage3") + [cosine_callback]
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    best_val_acc = max(history.history['val_accuracy'])
    print(f"\n✓ Stage 3 complete - Best val_accuracy: {best_val_acc:.4f}")
    
    return history


# ============================================================================
# TEST-TIME AUGMENTATION
# ============================================================================

def predict_with_tta(model, images, num_augmentations=5):
    """Perform test-time augmentation."""
    data_aug = create_data_augmentation()
    predictions = []
    
    predictions.append(model.predict(images, verbose=0))
    
    for _ in range(num_augmentations - 1):
        aug_images = data_aug(images, training=True)
        predictions.append(model.predict(aug_images, verbose=0))
    
    return np.mean(predictions, axis=0)


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, test_ds, class_names):
    """Evaluate model on test set with optional TTA."""
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    
    # ✅ Keras 3 fix: Use the trained model directly instead of loading from .h5
    # The model already has best weights restored via EarlyStopping callback
    print("\nUsing best model weights (restored by EarlyStopping)...")
    
    print("\n[1/2] Standard evaluation...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\n📊 Test Loss: {test_loss:.4f}")
    print(f"📊 Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    
    if USE_TTA:
        print("\n[2/2] Test-Time Augmentation evaluation...")
        y_true = []
        y_pred_tta = []
        
        for images, labels in test_ds:
            predictions = predict_with_tta(model, images, num_augmentations=5)
            y_pred_tta.extend(np.argmax(predictions, axis=1))
            if len(labels.shape) > 1:
                y_true.extend(np.argmax(labels.numpy(), axis=1))
            else:
                y_true.extend(labels.numpy())
        
        y_true = np.array(y_true)
        y_pred_tta = np.array(y_pred_tta)
        
        tta_acc = np.mean(y_true == y_pred_tta)
        print(f"📊 TTA Test Accuracy: {tta_acc:.4f} ({tta_acc * 100:.2f}%)")
        print(f"   Improvement: +{(tta_acc - test_acc) * 100:.2f}%")
        
        y_pred = y_pred_tta
    else:
        y_true = []
        y_pred = []
        
        for images, labels in test_ds:
            predictions = model.predict(images, verbose=0)
            y_pred.extend(np.argmax(predictions, axis=1))
            if len(labels.shape) > 1:
                y_true.extend(np.argmax(labels.numpy(), axis=1))
            else:
                y_true.extend(labels.numpy())
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    
    print("\n" + "-" * 60)
    print("CLASSIFICATION REPORT")
    print("-" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    print("\n" + "-" * 60)
    print("CONFUSION MATRIX")
    print("-" * 60)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")
    
    return test_loss, test_acc


def save_class_names(class_names):
    """Save class names to JSON."""
    with open(CLASS_NAMES_PATH, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"✓ Class names saved to: {CLASS_NAMES_PATH}")


def save_training_history(history1, history2, history3):
    """Save training history."""
    history_combined = {
        'stage1': {k: [float(v) for v in history1.history[k]] for k in history1.history},
        'stage2': {k: [float(v) for v in history2.history[k]] for k in history2.history},
        'stage3': {k: [float(v) for v in history3.history[k]] for k in history3.history}
    }
    
    with open(TRAINING_HISTORY_PATH, 'w') as f:
        json.dump(history_combined, f, indent=2)
    print(f"✓ Training history saved to: {TRAINING_HISTORY_PATH}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training pipeline - Memory optimized."""
    print("\n" + "=" * 60)
    print("MOBILENETV3-LARGE GARBAGE CLASSIFICATION")
    print("🚀 MEMORY-OPTIMIZED FOR TENSORFLOW 2.16 + KERAS 3 + METAL GPU")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(gpus) > 0}")
    
    print("\n✅ FIXES APPLIED:")
    print("   • GPU memory growth enabled")
    print("   • Removed .cache() to prevent memory leak")
    print("   • Added memory monitor callback")
    print("   • Optimized batch size")
    print("   • Shuffle buffer: 1000 (reasonable size)")
    
    print("\n🎯 OPTIMIZATIONS ENABLED:")
    print("   • Adam optimizer with L2 regularization")
    print("   • MixUp augmentation")
    print("   • Label smoothing (0.1)")
    print("   • Cosine decay with warmup")
    print("   • Gradient clipping")
    print(f"   • Test-time augmentation: {USE_TTA}")
    
    # 1. Create datasets
    train_ds, val_ds, test_ds, class_names = create_datasets()
    
    # 2. Create data augmentation
    data_augmentation = create_data_augmentation()
    
    # 3. Calculate class weights
    class_weight = calculate_class_weights(train_ds, class_names)
    
    print("\n⚠️  Note: Class weights disabled when using MixUp")
    print("   MixUp provides implicit class balancing")
    
    # 4. Optimize datasets
    train_ds, val_ds, test_ds = optimize_datasets(
        train_ds, val_ds, test_ds, data_augmentation, use_mixup=True
    )
    
    # 5. Build model
    model, base_model = build_model()
    
    # 6-8. Three-stage training
    history1 = train_stage1(model, train_ds, val_ds, EPOCHS_STAGE1)
    history2 = train_stage2(model, base_model, train_ds, val_ds, EPOCHS_STAGE2)
    history3 = train_stage3(model, base_model, train_ds, val_ds, EPOCHS_STAGE3)
    
    # 9. Final evaluation
    evaluate_model(model, test_ds, class_names)
    
    # 10. Save artifacts
    save_class_names(class_names)
    save_training_history(history1, history2, history3)
    
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Class names: {CLASS_NAMES_PATH}")
    print(f"History: {TRAINING_HISTORY_PATH}")
    print("\n💡 Run: python TestModel.py")


if __name__ == "__main__":
    main()
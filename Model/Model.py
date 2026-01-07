
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# Configuration
DATASET_DIR = "garbage_classification"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Load model
print("Loading model...")
model = load_model('resnet50v2_garbage_classifier.keras')
print("Model loaded successfully!")

# Create test dataset (same split as training)
print("\nLoading test dataset...")
val_test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode='int'
)

# Get class names and split to test set
class_names = val_test_ds.class_names
val_batches = int(len(val_test_ds) * 0.5)
test_ds = val_test_ds.skip(val_batches)

print(f"Found {len(class_names)} classes: {class_names}")
print(f"Test batches: {len(test_ds)}")

# Evaluate
print("\n" + "=" * 60)
print("Evaluating Model on Test Set")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Detailed predictions
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

print("\n" + "-" * 60)
print("Classification Report:")
print("-" * 60)
# Get unique labels in test set
unique_labels = sorted(set(y_true))
test_class_names = [class_names[i] for i in unique_labels]
print(classification_report(y_true, y_pred, labels=unique_labels, target_names=test_class_names))

print("\n" + "-" * 60)
print("Confusion Matrix:")
print("-" * 60)
print(confusion_matrix(y_true, y_pred))
import os
import itertools
import random
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array , load_img
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense
from tensorflow.keras.optimizers import Adam
import collections
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import pandas as pd
set_global_policy('mixed_float16')
base_dir = "RIDB_FORMATED"

num_individuals = 20
images_per_person = 5
def generate_image_paths(person_id):
    person_folder = os.path.join(base_dir, f"Person_{person_id}")
    image_paths = []
    for i in range(1, images_per_person + 1):
        image_name = f"IM{str(i).zfill(6)}_{person_id}.JPG"
        full_path = os.path.join(person_folder, image_name)
        image_paths.append(full_path)
    return image_paths

# Generate all unique positive pairs from the dataset
all_positive_pairs = []
for person in range(1, num_individuals + 1):
    images = generate_image_paths(person)
    pairs = list(itertools.combinations(images, 2))
    all_positive_pairs.extend(pairs)

# Generate all negative pairs from the dataset
all_negative_pairs_train = []
all_person_images = {person: generate_image_paths(person) for person in range(1, num_individuals + 1)}
for person1, person2 in itertools.combinations(range(1, num_individuals - 4), 2):
    for img1 in all_person_images[person1]:
        for img2 in all_person_images[person2]:
            all_negative_pairs_train.append((img1, img2))


all_negative_pairs_test = []
all_person_images = {person: generate_image_paths(person) for person in range(1, num_individuals + 1)}
for person1, person2 in itertools.combinations(range(16, num_individuals + 1 ), 2):
    for img1 in all_person_images[person1]:
        for img2 in all_person_images[person2]:
            all_negative_pairs_test.append((img1, img2))

# Balanced Negative Sampling for Training and Testing
def balanced_negative_sampling(all_negative_pairs, num_samples):
    selected_negatives = random.sample(all_negative_pairs, min(num_samples, len(all_negative_pairs)))
    random.shuffle(selected_negatives)
    return selected_negatives

train_positive = all_positive_pairs[:150]
test_positive = all_positive_pairs[150:200]

train_negative = balanced_negative_sampling(all_negative_pairs_train, 150)
test_negative = balanced_negative_sampling(all_negative_pairs_test, 50)
final_train_positive = train_positive
final_train_negative = train_negative

final_train_pairs = final_train_positive + final_train_negative
final_train_labels = [1] * len(final_train_positive) + [0] * len(final_train_negative)

final_test_pairs = test_positive + test_negative
final_test_labels = [1] * len(test_positive) + [0] * len(test_negative)

combined_train = list(zip(final_train_pairs, final_train_labels))
random.shuffle(combined_train)
final_train_pairs, final_train_labels = zip(*combined_train)

combined_test = list(zip(final_test_pairs, final_test_labels))
random.shuffle(combined_test)
final_test_pairs, final_test_labels = zip(*combined_test)
print("Final Training Set:")
print("Total pairs:", len(final_train_pairs))
print("Positive pairs:", final_train_labels.count(1))
print("Negative pairs:", final_train_labels.count(0))
print("\nFinal Test Set:")
print("Total pairs:", len(final_test_pairs))
print("Positive pairs:", final_test_labels.count(1))
print("Negative pairs:", final_test_labels.count(0))
train_pairs = final_train_pairs
train_labels = final_train_labels
test_pairs = final_test_pairs
test_labels = final_test_labels
def extract_person_id(filepath):
    dir_name = os.path.basename(os.path.dirname(filepath))
    if dir_name.startswith("Person_"):
        return dir_name.split("_")[1]
    else:
        return None

# Count positive pairs (label==1) where the person ID in image1 differs from image2
mismatch_count = 0
for pair, label in zip(test_pairs, test_labels):
    if label == 1:
        person1 = extract_person_id(pair[0])
        person2 = extract_person_id(pair[1])
        if person1 != person2:
            mismatch_count += 1

print("Number of label=1 pairs with different person IDs:", mismatch_count)


# Count negative pairs (label == 0) where the person ID in both images is the same
same_person_negative_count = 0
for pair, label in zip(test_pairs, test_labels):
    if label == 0:
        person1 = extract_person_id(pair[0])
        person2 = extract_person_id(pair[1])
        if person1 == person2:
            same_person_negative_count += 1

print("Number of label=0 pairs with the same person ID:", same_person_negative_count)
# Count the occurrences of each pair in the training set.
train_pair_counts = collections.Counter(train_pairs)
train_duplicate_pairs = {pair: count for pair, count in train_pair_counts.items() if count > 1}

if train_duplicate_pairs:
    print("Repeated pairs in the training set:")
    for pair, count in train_duplicate_pairs.items():
        print(f"{pair}: {count} times")
else:
    print("No repeated pairs in the training set.")

# Count the occurrences of each pair in the test set.
test_pair_counts = collections.Counter(test_pairs)
test_duplicate_pairs = {pair: count for pair, count in test_pair_counts.items() if count > 1}

if test_duplicate_pairs:
    print("Repeated pairs in the test set:")
    for pair, count in test_duplicate_pairs.items():
        print(f"{pair}: {count} times")
else:
    print("No repeated pairs in the test set.")

# Check for pairs that appear in both train and test sets.
common_pairs = set(train_pairs) & set(test_pairs)
if common_pairs:
    print("Pairs present in both train and test sets:")
    for pair in common_pairs:
        print(pair)
else:
    print("No pairs are present in both train and test sets.")

def plot_image_pairs(pairs, num_samples=5, title="Sample Image Pairs"):
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 2))
    fig.suptitle(title, fontsize=16)
    for i in range(num_samples):
        img1_path, img2_path = pairs[i]
        img1 = load_img(img1_path, target_size=(224, 224))
        img2 = load_img(img2_path, target_size=(224, 224))
        axes[i, 0].imshow(img1)
        axes[i, 0].axis("off")
        axes[i, 0].set_title("Image 1")
        axes[i, 1].imshow(img2)
        axes[i, 1].axis("off")
        axes[i, 1].set_title("Image 2")
    plt.tight_layout()
    plt.show()

test_pairs_list = list(test_pairs)
test_labels_list = list(test_labels)

positive_pairs = [pair for pair, label in zip(test_pairs_list, test_labels_list) if label == 1]
negative_pairs = [pair for pair, label in zip(test_pairs_list, test_labels_list) if label == 0]

print("Total positive test pairs:", len(positive_pairs))
print("Total negative test pairs:", len(negative_pairs))

sample_positive = random.sample(positive_pairs, 5)
sample_negative = random.sample(negative_pairs, 5)

plot_image_pairs(sample_positive, num_samples=5, title="Sample Positive Test Pairs")
plot_image_pairs(sample_negative, num_samples=5, title="Sample Negative Test Pairs")

def load_and_preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img / 255.0
    return img

def load_pair(pair):
    img1_path, img2_path = pair
    img1 = load_and_preprocess_image(img1_path)
    img2 = load_and_preprocess_image(img2_path)
    return img1, img2

def create_dataset(pairs, labels):
    imgs1, imgs2 = [], []
    for pair in pairs:
        image1, image2 = load_pair(pair)
        imgs1.append(image1)
        imgs2.append(image2)
    return np.array(imgs1), np.array(imgs2), np.array(labels)

train_img1, train_img2, train_labels_np = create_dataset(train_pairs, train_labels)
test_img1, test_img2, test_labels_np = create_dataset(test_pairs, test_labels)

print("Training samples:", train_img1.shape, train_img2.shape, train_labels_np.shape)
print("Testing samples:", test_img1.shape, test_img2.shape, test_labels_np.shape)
def euclidean_distance(vectors):
    vector1, vector2 = vectors
    sum_square = tf.reduce_sum(tf.square(vector1 - vector2), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))
def embedding_network(in_shape):
    input_layer = keras.layers.Input(in_shape)
    x = keras.layers.Conv2D(128, (5, 5), activation="relu")(input_layer)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Conv2D(32, (3, 3), activation="relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    return keras.Model(inputs=input_layer, outputs=x, name="base_network")


def SiameseNetwork(in_shape):
    input_1 = Input(shape=in_shape)
    input_2 = Input(shape=in_shape)
    embedding_net_obj = embedding_network(in_shape)
    twin_1 = embedding_net_obj(input_1)
    twin_2 = embedding_net_obj(input_2)
    merge_layer = Lambda(euclidean_distance, output_shape=(1,))([twin_1, twin_2])
    dense_layer = Dense(128, activation="relu")(merge_layer)
    output_layer = Dense(1, activation="sigmoid", dtype="float32")(dense_layer)
    
    return keras.Model(inputs=[input_1, input_2], outputs=output_layer)



input_shape = (224, 224, 3)
siamese_net = SiameseNetwork(input_shape)
siamese_net.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=Adam(learning_rate=2e-3),
    metrics=['accuracy']
)
siamese_net.summary()
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_path = "best_siamese_model.h5"

checkpoint_callback = ModelCheckpoint(
    checkpoint_path,
    monitor="val_accuracy",  
    save_best_only=True,     
    save_weights_only=True,  
    mode="max",               
    verbose=1
)
history = siamese_net.fit(
    [train_img1, train_img2],
    train_labels_np,
    validation_data=([test_img1, test_img2], test_labels_np),
    epochs=30,
    batch_size=8,
    callbacks=[checkpoint_callback]
)
siamese_net.load_weights(checkpoint_path)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curves")

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Curves")
plt.tight_layout()
plt.show()
test_predictions = siamese_net.predict([test_img1, test_img2])
test_predictions = test_predictions.flatten()

threshold = 0.5
test_pred_labels = (test_predictions > threshold).astype(int)
cm = confusion_matrix(test_labels_np, test_pred_labels)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=["Predicted Negative", "Predicted Positive"],
            yticklabels=["Actual Negative", "Actual Positive"])
plt.title("Confusion Matrix")
plt.xlabel("Prediction")
plt.ylabel("Ground Truth")
plt.show()

report = classification_report(test_labels_np, test_pred_labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()

plt.figure(figsize=(8, 4))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="Greens", fmt=".2f")
plt.title("Classification Report")
plt.show()
def authenticate_user(user_id, input_image_path, threshold=0.5):
    ref_path = generate_image_paths(user_id)[0]

    ref_img = load_and_preprocess_image(ref_path)
    input_img = load_and_preprocess_image(input_image_path)

    ref_img_exp = np.expand_dims(ref_img, axis=0)
    input_img_exp = np.expand_dims(input_img, axis=0)

    similarity = siamese_net.predict([ref_img_exp, input_img_exp])
    similarity_score = similarity[0][0]

    authenticated = similarity_score > threshold

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ref_disp = load_img(ref_path, target_size=(224, 224))
    input_disp = load_img(input_image_path, target_size=(224, 224))
    axes[0].imshow(ref_disp)
    axes[0].set_title("Reference Image")
    axes[0].axis("off")
    axes[1].imshow(input_disp)
    axes[1].set_title("Input Image")
    axes[1].axis("off")
    plt.suptitle("Biometric Authentication")
    plt.show()

    print(f"User ID: {user_id}")
    print(f"Reference image: {ref_path}")
    print(f"Input image: {input_image_path}")
    print(f"Similarity score: {similarity_score:.4f}")
    if authenticated:
        print("Authentication Successful: The user is verified.\n")
    else:
        print("Authentication Failed: The user is not verified.\n")

    return similarity_score, authenticated
# Example 1: Genuine authentication.
user_id = 1
input_image_genuine = os.path.join(base_dir, "Person_1", "IM000002_1.JPG")
print("Genuine Authentication Example:")
score, auth = authenticate_user(user_id, input_image_genuine, threshold=0.5)

# Example 2: Impostor authentication.
user_id = 1
input_image_impostor = os.path.join(base_dir, "Person_2", "IM000001_2.JPG")
print("Impostor Authentication Example:")
score, auth = authenticate_user(user_id, input_image_impostor, threshold=0.5)

# Example 3: Genuine authentication.
user_id = 20
input_image_impostor = os.path.join(base_dir, "Person_20", "IM000005_20.JPG")
print("Impostor Authentication Example:")
score, auth = authenticate_user(user_id, input_image_impostor, threshold=0.5)

# Example 4: Impostor authentication.
user_id = 10
input_image_impostor = os.path.join(base_dir, "Person_20", "IM000005_20.JPG")
print("Impostor Authentication Example:")
score, auth = authenticate_user(user_id, input_image_impostor, threshold=0.5)
import zipfile
import os
import random
import shutil
import re
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import json

zip_file_path = "test_hackathon.zip"
extracted_dir = "test_hackathon_extracted"

if not os.path.exists(extracted_dir):
    os.makedirs(extracted_dir)

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    for file in zip_ref.namelist():
        if not file.endswith('.DS_Store'):
            zip_ref.extract(file, extracted_dir)

print("Extraction completed.")

#-------------------------------------------------------

extracted_dir = "test_hackathon_extracted/test_hackathon"
train_dir = "train_data"
test_dir = "test_data"

for directory in [train_dir, test_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

folders = ["ACRA Search", "Bank Advice & Statement", "Contract", "Emails", "Invoice", "Letters", "Agreement"]

train_ratio = 0.8  # 80% for training, 20% for testing

for folder in folders:
    folder_path = os.path.join(extracted_dir, folder)
    
    train_subdir = os.path.join(train_dir, folder)
    test_subdir = os.path.join(test_dir, folder)
    for subdir in [train_subdir, test_subdir]:
        if not os.path.exists(subdir):
            os.makedirs(subdir)
    
    files = [file for file in os.listdir(folder_path) if file.endswith('.pdf')]
    random.shuffle(files)
    num_train_files = int(len(files) * train_ratio)
    
    # Split the files into train and test sets
    train_files = files[:num_train_files]
    test_files = files[num_train_files:]
    
    # Move train files to the train directory
    for file in train_files:
        src = os.path.join(folder_path, file)
        dst = os.path.join(train_subdir, file)
        shutil.copy(src, dst)
    
    # Move test files to the test directory
    for file in test_files:
        src = os.path.join(folder_path, file)
        dst = os.path.join(test_subdir, file)
        shutil.copy(src, dst)

print("Train and test data generation completed.")

#--------------------------------------------------------------------

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        pages = convert_from_path(pdf_path, 300) 
        for page in pages:
            text += pytesseract.image_to_string(page)
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return text

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^A-Za-z\s]', '', text)
    return text 

X_train = []
y_train = []
train_dir = "train_data"
for category in os.listdir(train_dir):
    category_dir = os.path.join(train_dir, category)
    for file_name in os.listdir(category_dir):
        file_path = os.path.join(category_dir, file_name)
        if file_name.endswith('.pdf'):
            txt_file_path = file_path + ".txt"
            if os.path.exists(txt_file_path):
                print(f"Skipping {file_name}, already processed.")
                with open(txt_file_path, 'r') as text_file:
                    preprocessed_text = text_file.read()
                X_train.append(preprocessed_text)
                y_train.append(category)
                continue
            print(f"Processing {file_path}...")
            text = extract_text_from_pdf(file_path)
            preprocessed_text = preprocess_text(text)
            with open(txt_file_path, 'w') as text_file:
                text_file.write(preprocessed_text)
            X_train.append(preprocessed_text)
            y_train.append(category)

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)

# print("Initializing SVM classifier...")
clf = SVC(kernel='linear', C=1.0)
# print("Training SVM classifier...")
clf.fit(X_train_tfidf, y_train)
# print("Predicting on validation set...")
y_pred_val = clf.predict(X_val_tfidf)

# Calculate validation accuracy
val_accuracy = accuracy_score(y_val, y_pred_val)
print("Validation Accuracy:", val_accuracy)

# Load and preprocess test data
X_test = []
y_test = []
test_dir = "test_data"
for category in os.listdir(test_dir):
    category_dir = os.path.join(test_dir, category)
    for file_name in os.listdir(category_dir):
        file_path = os.path.join(category_dir, file_name)
        if file_name.endswith('.pdf'):
            txt_file_path = file_path + ".txt"
            if os.path.exists(txt_file_path):
                print(f"Skipping {file_name}, already processed.")
                with open(txt_file_path, 'r') as text_file:
                    preprocessed_text = text_file.read()
                X_test.append(preprocessed_text)
                y_test.append(category)
                continue
            
            print(f"Processing {file_path}...")
            text = extract_text_from_pdf(file_path)
            preprocessed_text = preprocess_text(text)
            with open(txt_file_path, 'w') as text_file:
                text_file.write(preprocessed_text)
            X_test.append(preprocessed_text)
            y_test.append(category)

# Encode category names into numerical labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Transform test data using TF-IDF vectorizer
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Predict on test set
print("Predicting on test set...")
y_pred_test = clf.predict(X_test_tfidf)

# Convert predicted labels to integers using LabelEncoder
y_pred_test_encoded = label_encoder.transform(y_pred_test)

# Calculate test accuracy
test_accuracy = accuracy_score(y_test_encoded, y_pred_test_encoded)
print("Test Accuracy:", test_accuracy)

#----------------------------------------------------------

# Generate JSON output
output_json = []
for i, document_text in enumerate(X_test):
    X_test_tfidf = tfidf_vectorizer.transform([document_text])
    output_json.append({
        "file_name": f"{i+1}_document.pdf",
        "predicted_class": clf.predict(X_test_tfidf)[0],
        # "predicted_date": extract_date(document_text),  
        # "predicted_name": generate_document_name(document_text)  
    })

output_file_path = "output.json"
with open(output_file_path, "w") as json_file:
    json.dump({"files": output_json}, json_file, indent=4)

print("JSON output file saved.")

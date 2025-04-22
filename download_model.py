import os
import gdown
import zipfile
import shutil

# 👇 Replace this with your own Google Drive model link
MODEL_URL = "https://drive.google.com/file/d/1bU2-jB2XNM2e3XWCTW4nb7nq0S_LauIo/view?usp=drive_link"
TARGET_ZIP = "pretrained_model.zip"

# Create folder to store model if it doesn't exist
os.makedirs("pretrained_model", exist_ok=True)

print("⬇️ Downloading pretrained model...")
gdown.download(MODEL_URL, TARGET_ZIP, fuzzy=True,quiet=False)

print("📦 Extracting model...")
with zipfile.ZipFile(TARGET_ZIP, 'r') as zip_ref:
    zip_ref.extractall("pretrained_model")

inner_dir = os.path.join("pretrained_model", "pretrained_model")
if os.path.exists(inner_dir):
    for filename in os.listdir(inner_dir):
        shutil.move(os.path.join(inner_dir, filename), "pretrained_model")
    os.rmdir(inner_dir)

print("✅ Pretrained model is ready!")

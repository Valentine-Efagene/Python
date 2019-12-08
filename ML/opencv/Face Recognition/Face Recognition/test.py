import os

data_folder_path = "data/images"
dirs = os.listdir(data_folder_path)
faces = []
labels = []
    
for dir_name in dirs:
    
    if not dir_name.startswith("s"):
        continue

    label = int(dir_name.replace("s", ""))
    subject_dir_path = data_folder_path + "/" + dir_name
    subject_images_names = os.listdir(subject_dir_path)
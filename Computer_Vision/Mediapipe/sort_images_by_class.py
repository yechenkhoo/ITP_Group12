import os
import shutil
import pandas as pd

base_dir = 'datasets/roboflow' 
folders = ['test', 'train', 'valid']  

for folder in folders:
    csv_path = os.path.join(base_dir, folder, '_annotations.csv')
    
    df = pd.read_csv(csv_path)
    
    print(f'Processing folder: {folder}')
    
    for index, row in df.iterrows():
        filename = row['filename']
        class_name = row['class']
        
        # Create class folder (e.g., P4) if it doesnt exit
        class_folder = f'P{class_name}'
        class_folder_path = os.path.join(base_dir, folder, class_folder)
        if not os.path.exists(class_folder_path):
            os.makedirs(class_folder_path)
        
        #Source and destination paths
        src_file = os.path.join(base_dir, folder, filename)
        dest_file = os.path.join(class_folder_path, filename)
        
        if os.path.exists(src_file):
            shutil.move(src_file, dest_file)
    
    print(f"Images in {folder} have been sorted.")

print("All folders have been processed.")
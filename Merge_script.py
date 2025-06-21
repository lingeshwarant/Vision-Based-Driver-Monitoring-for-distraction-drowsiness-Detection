import os
import shutil


def merge_datasets(input_folders, output_folder):
    splits = ['train', 'valid', 'test']
    subdirs = ['images', 'labels']

    for split in splits:
        for subdir in subdirs:
            # Create destination subfolders if they don't exist
            dest_dir = os.path.join(output_folder, split, subdir)
            os.makedirs(dest_dir, exist_ok=True)

            for folder in input_folders:
                src_dir = os.path.join(folder, split, subdir)
                if not os.path.exists(src_dir):
                    print(f"Warning: {src_dir} not found.")
                    continue
                for file_name in os.listdir(src_dir):
                    src_file = os.path.join(src_dir, file_name)
                    dest_file = os.path.join(dest_dir, file_name)
                    if os.path.exists(dest_file):
                        # Rename to avoid collision
                        name, ext = os.path.splitext(file_name)
                        new_name = f"{name}_{os.path.basename(folder)}{ext}"
                        dest_file = os.path.join(dest_dir, new_name)
                    shutil.copy2(src_file, dest_file)
    print("✅ Merge completed successfully.")


# Define paths
input_folders = ['drowsiness_latest_yolov8', 'drowsiness_older_yolov8']
output_folder = 'Total_dataset'

# Run merge
merge_datasets(input_folders, output_folder)

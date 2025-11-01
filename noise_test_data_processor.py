#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Noise_test data processor
File path: /public_new/work_space/fengjiali/MKwithML/MFNversion_2/dual_branch/noise_test_data_processor.py

Dedicated to handling the Noise_test dataset and inherits from SimulateDataProcessor.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data_processor import SimulateDataProcessor


class NoiseTestDataProcessor(SimulateDataProcessor):
    """
    Processor for the Noise_test dataset.

    Dataset layout:
    Noise_test/
        1.00/
            begin_xxxx_end_xxxx/
                down_begin_xxxx_end_xxxx/
                    displacement.xlsx
                    image_001.npy
                    image_002.npy
                    ...
                up_begin_xxxx_end_xxxx/
                    ...
        1.05/
        ...
    """
    
    def __init__(self, dataset_root="/public_new/work_space/fengjiali/MKwithML/Noise_test"):
        # Call the parent constructor but skip extra logging because multiple noise levels are scanned
        self.dataset_root = dataset_root
        self.wavelength = 633.017  # λ = 633.017nm (HeNe laser)
        print(f"[Noise Test Data] Initialize noise test data processor: {dataset_root}")
    
    def get_noise_levels(self):
        """Return the list of available noise levels"""
        noise_levels = []
        
        if not os.path.exists(self.dataset_root):
            print(f"[Error] Dataset root does not exist: {self.dataset_root}")
            return []
        
        for item in os.listdir(self.dataset_root):
            item_path = os.path.join(self.dataset_root, item)
            if os.path.isdir(item_path):
                try:
                    # Attempt to parse the folder name as a floating-point noise level
                    noise_level = float(item)
                    noise_levels.append(noise_level)
                except ValueError:
                    continue
        
        # Sort ascending
        noise_levels.sort()
        return noise_levels
    
    def get_all_data_folders_by_noise_level(self, noise_level):
        """
        Return every valid folder for a given noise level.

        Args:
            noise_level: Noise level (for example 1.00, 1.05, ...).

        Returns:
            List of folder paths for that noise level.
        """
        all_folders = []
        
        # Build the folder path for this noise level
        noise_level_str = f"{noise_level:.2f}"
        noise_level_path = os.path.join(self.dataset_root, noise_level_str)
        
        if not os.path.exists(noise_level_path):
            print(f"[Warning] Noise level folder missing: {noise_level_path}")
            return []
        
        try:
            # Iterate through every begin_*_end_* folder
            for begin_folder in os.listdir(noise_level_path):
                if not begin_folder.startswith('begin_'):
                    continue
                
                begin_path = os.path.join(noise_level_path, begin_folder)
                if not os.path.isdir(begin_path):
                    continue
                
                # Inspect up_ and down_ subdirectories
                for subdir in os.listdir(begin_path):
                    if subdir.startswith(('up_', 'down_')):
                        subdir_path = os.path.join(begin_path, subdir)
                        if self._is_valid_data_folder(subdir_path):
                            all_folders.append(subdir_path)
        except Exception as e:
            print(f"[Error] Failed to scan folders: {e}")
        
        return all_folders
    
    def _is_valid_data_folder(self, folder_path):
        """Check whether a folder is a valid data leaf"""
        # Require displacement.xlsx and at least one image_*.npy file
        has_excel = os.path.exists(os.path.join(folder_path, 'displacement.xlsx'))
        has_images = any(f.startswith('image_') and f.endswith('.npy') 
                        for f in os.listdir(folder_path))
        return has_excel and has_images
    
    def process_single_folder(self, folder_path):
        """
        Process a single noise test data folder.

        Returns a dictionary with:
            - images_ch1, ch2, ch3: (T, H, W) image series
            - numerical_features: (T, D) numerical placeholders (zero filled)
            - displacements: (T,) absolute displacement labels in nanometers
            - orders: (T,) order labels (set to 0)
            - image_indices: (T,) image indices (matching the Excel sheet)
        """
        try:
            # Load displacement annotations
            excel_path = os.path.join(folder_path, 'displacement.xlsx')
            df = pd.read_excel(excel_path)
            
            # Noise_test should use the absolute displacement column instead of Δm
            if 'Displacement (nm)' in df.columns:
                displacements = df['Displacement (nm)'].values.astype(np.float32)
            else:
                print(f"[Warning] Missing 'Displacement (nm)' column: {folder_path} (Noise_test ignores Δm)")
                return None
            
            # Extract image indices
            image_indices = df['Image Index'].values
            
            # Load images using the Excel-provided ordering
            images = []
            valid_displacements = []
            valid_indices = []
            
            for idx, disp in zip(image_indices, displacements):
                # Image file names follow image_XXX.npy where XXX is zero padded to 3 digits
                img_filename = f"image_{idx:03d}.npy"
                img_path = os.path.join(folder_path, img_filename)
                
                if os.path.exists(img_path):
                    img = np.load(img_path)
                    images.append(img)
                    valid_displacements.append(disp)
                    valid_indices.append(idx)
                else:
                    print(f"[Warning] Missing image file: {img_path}")
            
            if len(images) == 0:
                print(f"[Warning] No valid image files detected: {folder_path}")
                return None
            
            images = np.array(images).astype(np.float32)  # (T, H, W)
            displacements = np.array(valid_displacements).astype(np.float32)
            image_indices = np.array(valid_indices).astype(np.int32)
            
            T = len(images)
            
            # Orders remain zero because evaluation focuses on displacement
            orders = np.zeros(T, dtype=np.int64)
            
            # Numerical placeholders (not used during testing)
            numerical_features = np.zeros((T, 7), dtype=np.float32)
            
            return {
                'images_ch1': images,  # (T, H, W)
                'images_ch2': images,  # Replicate single-channel data to three channels
                'images_ch3': images,  
                'numerical_features': numerical_features,
                'displacements': displacements,
                'orders': orders,
                'image_indices': image_indices,
                'folder_path': folder_path
            }
            
        except Exception as e:
            print(f"[Error] Failed to process folder {folder_path}: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    """Manual smoke test for the noise test processor"""
    print("=== Test noise test data processor ===")
    
    processor = NoiseTestDataProcessor()
    
    noise_levels = processor.get_noise_levels()
    print(f"\nFound {len(noise_levels)} noise levels:")
    print(noise_levels)
    
    if noise_levels:
        test_noise_level = noise_levels[0]
        print(f"\nEvaluating noise level: {test_noise_level}")
        
        folders = processor.get_all_data_folders_by_noise_level(test_noise_level)
        print(f"Found {len(folders)} data folders")
        
        if folders:
            print(f"\nProcessing first folder: {folders[0]}")
            data = processor.process_single_folder(folders[0])
            
            if data:
                print("\nData structure:")
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        print(f"  {key}: {value.shape}, dtype={value.dtype}")
                        if key == 'displacements':
                            print(f"    Range: [{value.min():.2f}, {value.max():.2f}] nm")
                    else:
                        print(f"  {key}: {value}")
    
    print("\n=== Tests finished ===")

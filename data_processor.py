#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MFNversion_2 data processor
File path: /public_new/work_space/fengjiali/MKwithML/MFNversion_2/dual_branch/data_processor.py

Supports:
1. Simulated data loading (Dataset_simulate) for pretraining
2. Real data loading (Dataset_finetune) for fine-tuning
3. Order label generation (map Δm to {-2, -1, 0, 1, 2})
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import warnings
import cv2
from PIL import Image
import torchvision.transforms as transforms
import math

warnings.filterwarnings('ignore')

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


# ==================== Base data processor ====================
class BaseDataProcessor:
    """Base class for data processors"""
    def __init__(self, dataset_root):
        self.dataset_root = dataset_root
        self.wavelength = 633.017  # λ = 633.017nm (HeNe laser)
        
    def get_all_data_folders(self, max_folders=None):
        """Return all data folders (leaf nodes: up_/down_)"""
        raise NotImplementedError
    
    def _is_valid_data_folder(self, folder_path):
        """Check whether the folder contains a valid data sample"""
        raise NotImplementedError
    
    def compute_order_and_displacement(self, delta_m):
        """
        Convert Δm into the discrete order label and displacement value.

        Args:
            delta_m: Δm value (can be fractional).

        Returns:
            order: Order label (integer 0-4 mapped from physical orders -2 to 2).
            displacement_nm: Displacement in the range [0, λ/2] measured in nanometers.

        Processing logic:
            If Δm < 0: physical_order = ceil(Δm)
            If Δm > 0: physical_order = floor(Δm)
            If Δm = 0: physical_order = 0

            Displacement: d = (λ/2) * (Δm - physical_order)

            Label mapping: order_label = physical_order + 2 (map -2~2 to 0~4).
        """
        if delta_m < 0:
            physical_order = math.ceil(delta_m)  # Round toward positive infinity (larger integer)
        elif delta_m > 0:
            physical_order = math.floor(delta_m)  # Round toward negative infinity (smaller integer)
        else:
            physical_order = 0
        
        # Compute displacement with d = (λ/2) * (Δm - physical_order)
        lambda_half = self.wavelength / 2  # 316.5085 nm
        displacement_nm = lambda_half * (delta_m - physical_order)
        
        # Clamp displacement within [0, λ/2]
        displacement_nm = max(0, min(displacement_nm, lambda_half))
        
        # Map physical orders {-2, -1, 0, 1, 2} to label range {0, 1, 2, 3, 4}
        order_label = physical_order + 2
        
        return order_label, displacement_nm
    
    def process_single_folder(self, folder_path):
        """Process a single folder and return the raw sample"""
        raise NotImplementedError


# ==================== Simulated data processor ====================
class SimulateDataProcessor(BaseDataProcessor):
    """Data processor for simulated datasets used in pretraining"""
    def __init__(self, dataset_root="/public_new/work_space/fengjiali/MKwithML/Dataset_simulate"):
        super().__init__(dataset_root)
        print(f"[Simulated Data] Initialize simulated data processor: {dataset_root}")
    
    def _is_valid_data_folder(self, folder_path):
        """Check whether a folder contains a valid simulated sample"""
        # Require displacement.xlsx and at least one image_*.npy
        has_excel = os.path.exists(os.path.join(folder_path, 'displacement.xlsx'))
        has_images = any(f.startswith('image_') and f.endswith('.npy') 
                        for f in os.listdir(folder_path))
        return has_excel and has_images
    
    def get_all_data_folders(self, max_folders=None):
        """Return all data folders (leaf nodes: up_/down_)"""
        all_folders = []
        
        if not os.path.exists(self.dataset_root):
            print(f"[Error] Dataset root does not exist: {self.dataset_root}")
            return []
        
        # Traverse every begin_*_end_* folder
        try:
            for begin_folder in os.listdir(self.dataset_root):
                if not begin_folder.startswith('begin_'):
                    continue
                
                begin_path = os.path.join(self.dataset_root, begin_folder)
                if not os.path.isdir(begin_path):
                    continue
                
                # Inspect up_ and down_ subdirectories
                for subdir in os.listdir(begin_path):
                    if subdir.startswith(('up_', 'down_')):
                        subdir_path = os.path.join(begin_path, subdir)
                        if self._is_valid_data_folder(subdir_path):
                            all_folders.append(subdir_path)
                            
                            if max_folders and len(all_folders) >= max_folders:
                                return all_folders
        except Exception as e:
            print(f"[Error] Failed to scan folders: {e}")
        
        return all_folders
    
    def process_single_folder(self, folder_path):
        """
        Process a single simulated data folder.

        Returns a dictionary containing:
            - images_ch1, ch2, ch3: (T, H, W) image sequences
            - numerical_features: (T, D) numerical features (zero padded)
            - displacements: (T,) ground-truth displacement in nanometers
            - orders: (T,) order labels (set to 0 for simulated data)
        """
        try:
            # Load displacement labels
            excel_path = os.path.join(folder_path, 'displacement.xlsx')
            df = pd.read_excel(excel_path)
            
            if 'Displacement (nm)' not in df.columns:
                print(f"[Warning] Missing 'Displacement (nm)' column: {folder_path}")
                return None
            
            displacements = df['Displacement (nm)'].values
            
            # Gather all image files
            image_files = sorted([f for f in os.listdir(folder_path) 
                                 if f.startswith('image_') and f.endswith('.npy')])
            
            if len(image_files) == 0:
                print(f"[Warning] No image files found: {folder_path}")
                return None
            
            T = min(len(image_files), len(displacements))
            
            # Load image tensors
            images = []
            for i in range(T):
                img_path = os.path.join(folder_path, image_files[i])
                img = np.load(img_path)
                images.append(img)
            
            images = np.array(images).astype(np.float32)  # (T, H, W)
            displacements = displacements[:T].astype(np.float32)
            
            # Simulated data uses zero order labels because pretraining only regresses displacement
            orders = np.zeros(T, dtype=np.int64)
            
            # Numerical placeholders (simulated data lacks auxiliary measurements)
            numerical_features = np.zeros((T, 7), dtype=np.float32)
            
            return {
                'images_ch1': images,  # (T, H, W)
                'images_ch2': images,  # Simulated data has a single channel, replicate three times
                'images_ch3': images,  
                'numerical_features': numerical_features,
                'displacements': displacements,
                'orders': orders,
                'folder_path': folder_path
            }
            
        except Exception as e:
            print(f"[Error] Failed to process folder {folder_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_all_data_for_batch(self, batch_folders):
        """Process a batch of folders"""
        batch_data_list = []
        
        for folder_path in batch_folders:
            data = self.process_single_folder(folder_path)
            if data is not None:
                batch_data_list.append(data)
        
        print(f"[Simulated Data] Processed {len(batch_data_list)}/{len(batch_folders)} folders successfully")
        return batch_data_list if batch_data_list else None


# ==================== Real data processor ====================
class FinetuneDataProcessor(BaseDataProcessor):
    """Data processor for fine-tuning on real measurements"""
    def __init__(self, dataset_root="/public_new/work_space/fengjiali/MKwithML/Dataset_finetune"):
        super().__init__(dataset_root)
        print(f"[Fine-tune Data] Initialize fine-tune data processor: {dataset_root}")
    
    def _is_valid_data_folder(self, folder_path):
        """Check whether the folder contains a valid fine-tuning sample"""
        required_files = ['Displacement_before_training.xlsx', 'voltages.xlsx']
        has_required = all(os.path.exists(os.path.join(folder_path, f)) for f in required_files)
        
        # Ensure there is at least one image file
        try:
            has_images = any(f.startswith('image_') and f.endswith('.npy') 
                           for f in os.listdir(folder_path))
        except:
            has_images = False
        
        return has_required and has_images
    
    def get_all_data_folders(self, max_folders=None):
        """Return all data folders directly under Dataset_finetune (up_/down_)"""
        all_folders = []
        
        if not os.path.exists(self.dataset_root):
            print(f"[Error] Dataset root does not exist: {self.dataset_root}")
            return []
        
        try:
            # Directly scan all folders inside Dataset_finetune
            for folder_name in os.listdir(self.dataset_root):
                if folder_name.startswith(('up_', 'down_')):
                    folder_path = os.path.join(self.dataset_root, folder_name)
                    if os.path.isdir(folder_path) and self._is_valid_data_folder(folder_path):
                        all_folders.append(folder_path)
                        
                        if max_folders and len(all_folders) >= max_folders:
                            return all_folders
        except Exception as e:
            print(f"[Error] Failed to scan folders: {e}")
        
        return all_folders
    
    def extract_image_features(self, img_array, prev_img=None, img_index=0, total_images=1):
        """Extract seven numerical features from an image sequence (see real_data_processor.py)"""
        try:
            # Ensure float32 dtype and normalization
            img = img_array.astype(np.float32)
            
            # Convert RGB images to grayscale
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = np.dot(img, [0.299, 0.587, 0.114])
            
            if len(img.shape) != 2:
                raise ValueError(f"Image should be 2D after conversion, but got shape: {img.shape}")
            
            if img.max() > 1.0:
                img = img / 255.0
            
            # Compute difference image if a previous frame is available
            if prev_img is not None:
                prev_img = prev_img.astype(np.float32)
                if len(prev_img.shape) == 3 and prev_img.shape[2] == 3:
                    prev_img = np.dot(prev_img, [0.299, 0.587, 0.114])
                if prev_img.max() > 1.0:
                    prev_img = prev_img / 255.0
                diff_img = np.abs(img - prev_img)
            else:
                diff_img = np.zeros_like(img)
            
            numerical_features = []
            
            # (1) Normalized frame index
            normalized_index = img_index / (total_images - 1) if total_images > 1 else 0
            numerical_features.append(normalized_index)
            
            # (2) Mean intensity
            mean_intensity = np.mean(img)
            numerical_features.append(mean_intensity)
            
            # (3) Intensity variance
            intensity_variance = np.var(img)
            numerical_features.append(intensity_variance)
            
            # (4) Number of stripe peaks
            try:
                from scipy.signal import find_peaks
                horizontal_projection = np.mean(img, axis=0)
                if len(horizontal_projection.shape) != 1:
                    horizontal_projection = horizontal_projection.flatten()
                
                if len(horizontal_projection) == 0:
                    num_peaks = 0
                else:
                    mean_val = np.mean(horizontal_projection)
                    if mean_val > 0:
                        peaks, _ = find_peaks(horizontal_projection, height=mean_val)
                        num_peaks = len(peaks)
                    else:
                        num_peaks = 0
            except Exception as e:
                num_peaks = 0
            numerical_features.append(num_peaks)
            
            # (5) Low-frequency energy ratio in the spectrum
            try:
                from scipy.fft import fft2, fftshift
                fft_result = fft2(img)
                fft_magnitude = np.abs(fftshift(fft_result))
                
                rows, cols = fft_magnitude.shape
                center_row, center_col = rows // 2, cols // 2
                low_freq_radius = min(rows, cols) // 8
                y, x = np.ogrid[:rows, :cols]
                mask = (x - center_col)**2 + (y - center_row)**2 <= low_freq_radius**2
                low_freq_energy = np.sum(fft_magnitude[mask]**2)
                total_energy = np.sum(fft_magnitude**2)
                low_freq_ratio = low_freq_energy / total_energy if total_energy > 0 else 0
            except Exception as e:
                low_freq_ratio = 0
            numerical_features.append(low_freq_ratio)
            
            # (6) Spectral centroid frequency
            try:
                from scipy.fft import fft2, fftshift
                if 'fft_magnitude' not in locals():
                    fft_result = fft2(img)
                    fft_magnitude = np.abs(fftshift(fft_result))
                
                rows, cols = fft_magnitude.shape
                center_col = cols // 2
                center_row = rows // 2
                total_magnitude = np.sum(fft_magnitude)
                if total_magnitude > 0:
                    center_freq_x = np.sum(np.arange(cols) * np.sum(fft_magnitude, axis=0)) / total_magnitude
                    center_freq_y = np.sum(np.arange(rows) * np.sum(fft_magnitude, axis=1)) / total_magnitude
                    center_frequency = np.sqrt((center_freq_x - center_col)**2 + (center_freq_y - center_row)**2)
                else:
                    center_frequency = 0
            except Exception as e:
                center_frequency = 0
            numerical_features.append(center_frequency)
            
            # (7) Image entropy
            try:
                hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 1))
                hist_normalized = hist / np.sum(hist)
                entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
                if np.isnan(entropy) or np.isinf(entropy):
                    entropy = 0
            except Exception as e:
                entropy = 0
            numerical_features.append(entropy)
            
            return np.array(numerical_features, dtype=np.float32)
            
        except Exception as e:
            print(f"[Error] Failed to extract image features: {e}")
            return np.zeros(7, dtype=np.float32)
    
    def process_single_folder(self, folder_path):
        """Process a single fine-tuning data folder"""
        try:
            # Load displacement and Δm data
            disp_excel_path = os.path.join(folder_path, 'Displacement_before_training.xlsx')
            df_disp = pd.read_excel(disp_excel_path)
            
            if 'd(nm)' not in df_disp.columns or 'Δm' not in df_disp.columns:
                print(f"[Warning] Missing 'd(nm)' or 'Δm' column: {folder_path}")
                return None
            
            delta_m_values = df_disp['Δm'].values
            
            # Convert Δm to order labels and displacements
            orders = []
            displacements = []
            for delta_m in delta_m_values:
                order, displacement = self.compute_order_and_displacement(delta_m)
                orders.append(order)
                displacements.append(displacement)
            
            orders = np.array(orders, dtype=np.int64)
            displacements = np.array(displacements, dtype=np.float32)
            
            # Load image tensors
            image_files = sorted([f for f in os.listdir(folder_path) 
                                 if f.startswith('image_') and f.endswith('.npy')])
            
            if len(image_files) == 0:
                print(f"[Warning] No image files found: {folder_path}")
                return None
            
            T = min(len(image_files), len(orders))
            
            images = []
            numerical_features_list = []
            prev_img = None
            
            for i in range(T):
                img_path = os.path.join(folder_path, image_files[i])
                img = np.load(img_path)
                
                # Normalize image dimensionality to single channel
                if img.ndim == 3:
                    # Image has shape (H, W, 3); use the first channel as the grayscale image
                    img = img[:, :, 0]
                elif img.ndim != 2:
                    print(f"[Warning] Unexpected image shape: {img.shape}, path: {img_path}")
                    continue
                
                images.append(img)
                
                # Extract seven numerical features
                features = self.extract_image_features(img, prev_img, i, T)
                numerical_features_list.append(features)
                
                prev_img = img
            
            if len(images) == 0:
                print(f"[Warning] No valid images detected: {folder_path}")
                return None
            
            images = np.array(images).astype(np.float32)  # (T, H, W)
            numerical_features = np.array(numerical_features_list).astype(np.float32)  # (T, 7)
            
            # Ensure consistent lengths
            T = len(images)
            orders = orders[:T]
            displacements = displacements[:T]
            
            return {
                'images_ch1': images,  # Dataset_finetune images are single channel replicated three times
                'images_ch2': images,
                'images_ch3': images,
                'numerical_features': numerical_features,  # Contains the seven correctly extracted features
                'displacements': displacements,
                'orders': orders,
                'folder_path': folder_path
            }
            
        except Exception as e:
            print(f"[Error] Failed to process folder {folder_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_all_data_for_batch(self, batch_folders):
        """Process a batch of folders"""
        batch_data_list = []
        
        for folder_path in batch_folders:
            data = self.process_single_folder(folder_path)
            if data is not None:
                batch_data_list.append(data)
        
        print(f"[Fine-tune Data] Processed {len(batch_data_list)}/{len(batch_folders)} folders successfully")
        return batch_data_list if batch_data_list else None


# Backward compatibility alias
CleanedDataProcessor = FinetuneDataProcessor


# ==================== Dataset class ====================
class MFNDataset(Dataset):
    """
    MFN dataset abstraction

    Supports both pretraining and fine-tuning modes.
    """
    def __init__(self, processed_data_list, seq_len_min=2, seq_len_max=5, 
                 seq_step=1, mode='pretrain', add_noise=False, noise_std=0.01):
        """
        Args:
            processed_data_list: List of preprocessed samples.
            seq_len_min, seq_len_max: Sequence length bounds.
            seq_step: Sliding window step when sampling sequences.
            mode: Either 'pretrain' or 'finetune'.
            add_noise: Whether to inject noise (available during pretraining).
            noise_std: Standard deviation of the injected noise.
        """
        self.processed_data_list = processed_data_list
        self.seq_len_min = seq_len_min
        self.seq_len_max = seq_len_max
        self.seq_step = seq_step
        self.mode = mode
        self.add_noise = add_noise
        self.noise_std = noise_std
        
        # Prepare sequence indices
        self.sequence_indices = self._generate_sequence_indices()
        
        # Image preprocessing pipelines
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.mobilevit_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _generate_sequence_indices(self):
        """Enumerate every possible sequence index tuple"""
        indices = []
        
        for data_idx, data in enumerate(self.processed_data_list):
            T = data['images_ch1'].shape[0]
            
            for seq_len in range(self.seq_len_min, min(self.seq_len_max + 1, T + 1)):
                for start_idx in range(0, T - seq_len + 1, self.seq_step):
                    indices.append((data_idx, start_idx, seq_len))
        
        return indices
    
    def __len__(self):
        return len(self.sequence_indices)
    
    def __getitem__(self, idx):
        data_idx, start_idx, seq_len = self.sequence_indices[idx]
        data = self.processed_data_list[data_idx]
        
        end_idx = start_idx + seq_len
        
        # Retrieve the final frame in the selected sequence
        img_ch1 = data['images_ch1'][end_idx - 1]  # (H, W)
        img_ch2 = data['images_ch2'][end_idx - 1]  # (H, W)
        img_ch3 = data['images_ch3'][end_idx - 1]  # (H, W)
        
        # Ensure a 2D tensor
        if img_ch1.ndim != 2:
            raise ValueError(f"Image should be 2D, got shape {img_ch1.shape}")
        
        # Optional noise injection
        if self.add_noise:
            noise = np.random.normal(0, self.noise_std, img_ch1.shape).astype(np.float32)
            img_ch1 = img_ch1 + noise
            img_ch2 = img_ch2 + noise
            img_ch3 = img_ch3 + noise
        
        # Normalize to [0, 1]
        img_ch1 = np.clip(img_ch1 / 255.0, 0, 1)
        img_ch2 = np.clip(img_ch2 / 255.0, 0, 1)
        img_ch3 = np.clip(img_ch3 / 255.0, 0, 1)
        
        # Convert to tensor (C, H, W)
        img_ch1_tensor = torch.from_numpy(img_ch1).unsqueeze(0).float()
        img_ch2_tensor = torch.from_numpy(img_ch2).unsqueeze(0).float()
        img_ch3_tensor = torch.from_numpy(img_ch3).unsqueeze(0).float()
        
        # Standardize
        img_ch1_tensor = (img_ch1_tensor - 0.5) / 0.5
        img_ch2_tensor = (img_ch2_tensor - 0.5) / 0.5
        img_ch3_tensor = (img_ch3_tensor - 0.5) / 0.5
        
        # Build the three-channel view for MobileViT
        img_3channel = np.stack([img_ch1, img_ch2, img_ch3], axis=2)  # (H, W, 3)
        
        if img_3channel.ndim != 3 or img_3channel.shape[2] != 3:
            raise ValueError(f"3-channel image should be (H, W, 3), got shape {img_3channel.shape}")
        
        # Convert to PIL image and apply MobileViT transforms
        img_3channel_pil = Image.fromarray((img_3channel * 255).astype(np.uint8))
        img_3channel_tensor = self.mobilevit_transform(img_3channel_pil)
        
        # Retrieve the numerical sequence
        numerical_seq = data['numerical_features'][start_idx:end_idx]
        numerical_seq_tensor = torch.from_numpy(numerical_seq).float()
        
        # Compute displacement labels as differences between consecutive frames
        displacements = data['displacements']
        idx = end_idx - 1

        if self.mode == 'pretrain':
            if idx >= 1:
                displacement_label = float(displacements[idx] - displacements[idx - 1])
            else:
                # First frame has no previous observation; return NaN to be masked during training
                displacement_label = float('nan')

            return (img_ch1_tensor, img_ch2_tensor, img_ch3_tensor, img_3channel_tensor,
                   numerical_seq_tensor, displacement_label, seq_len)
        else:
            if idx >= 1:
                displacement_label = float(displacements[idx] - displacements[idx - 1])
            else:
                displacement_label = float('nan')

            order = data['orders'][idx]
            return (img_ch1_tensor, img_ch2_tensor, img_ch3_tensor, img_3channel_tensor,
                   numerical_seq_tensor, displacement_label, order, seq_len)


# ==================== Collate function ====================
def mfn_collate_fn(batch, mode='pretrain'):
    """
    Collate variable-length sequences into mini-batches.

    Returns:
        Pretraining mode: (img_ch1, img_ch2, img_ch3, img_3ch, num_seq, labels, lengths)
        Fine-tuning mode: (img_ch1, img_ch2, img_ch3, img_3ch, num_seq, disp_labels, order_labels, lengths)
    """
    if mode == 'pretrain':
        img_ch1, img_ch2, img_ch3, img_3ch, num_seq, labels, lengths = zip(*batch)
    else:
        img_ch1, img_ch2, img_ch3, img_3ch, num_seq, disp_labels, order_labels, lengths = zip(*batch)
    
    # Stack image tensors
    img_ch1 = torch.stack(img_ch1, dim=0)
    img_ch2 = torch.stack(img_ch2, dim=0)
    img_ch3 = torch.stack(img_ch3, dim=0)
    img_3ch = torch.stack(img_3ch, dim=0)
    
    # Pad numerical sequences
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    max_len = max(lengths)
    batch_size = len(num_seq)
    num_features = num_seq[0].shape[1]
    
    num_seq_padded = torch.zeros(batch_size, max_len, num_features)
    for i, seq in enumerate(num_seq):
        seq_len = seq.shape[0]
        num_seq_padded[i, :seq_len, :] = seq
    
    # Derive a mask that identifies valid displacement targets (True means valid)
    if mode == 'pretrain':
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        mask_tensor = ~torch.isnan(labels_tensor)
        return img_ch1, img_ch2, img_ch3, img_3ch, num_seq_padded, labels_tensor, mask_tensor, lengths_tensor
    else:
        disp_labels_tensor = torch.tensor(disp_labels, dtype=torch.float32)
        order_labels_tensor = torch.tensor(order_labels, dtype=torch.long)
        mask_tensor = ~torch.isnan(disp_labels_tensor)
        return img_ch1, img_ch2, img_ch3, img_3ch, num_seq_padded, disp_labels_tensor, order_labels_tensor, mask_tensor, lengths_tensor


if __name__ == "__main__":
    """Manual smoke test for the data processors"""
    print("=== Test simulated data processor ===")
    
    sim_processor = SimulateDataProcessor()
    sim_folders = sim_processor.get_all_data_folders(max_folders=2)
    
    if sim_folders:
        print(f"Found {len(sim_folders)} simulated data folders")
        sim_data = sim_processor.process_all_data_for_batch(sim_folders)
        
        if sim_data:
            print(f"\nShapes for the first folder:")
            for key, value in sim_data[0].items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {value}")
    
    print("\n=== Test fine-tune data processor ===")
    
    finetune_processor = FinetuneDataProcessor()
    finetune_folders = finetune_processor.get_all_data_folders(max_folders=2)
    
    if finetune_folders:
        print(f"Found {len(finetune_folders)} fine-tune data folders")
        finetune_data = finetune_processor.process_all_data_for_batch(finetune_folders)
        
        if finetune_data:
            print(f"\nInspecting the first folder:")
            for key, value in finetune_data[0].items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: {value.shape}")
                    if key == 'orders':
                        print(f"    Order range: {value.min()} to {value.max()}")
                        print(f"    Order histogram: {np.bincount(value + 2)}")  # +2 because minimum may be -2
                    elif key == 'displacements':
                        print(f"    Displacement range: {value.min():.2f} to {value.max():.2f} nm")
                else:
                    print(f"  {key}: {value}")
            
            dataset = MFNDataset(finetune_data, mode='finetune')
            print(f"\nDataset created with {len(dataset)} samples")
            
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"\nSample contains {len(sample)} elements")
                print(f"Order label: {sample[-2]}")
    
    print("\n=== Tests finished ===")

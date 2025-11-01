#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Noise evaluation script - assess pretrained model performance across noise levels
Original path: /public_new/work_space/fengjiali/MKwithML/MFNversion_2/dual_branch/test_noise.py

Capabilities:
1. Compute mean/maximum/minimum RMSE and standard deviation for each noise level
2. Measure inference speed per noise level (pure forward pass time, excluding data processing)
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import time
from datetime import datetime
import json
import warnings

warnings.filterwarnings('ignore')

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from model_definition import create_mfn_model
from noise_test_data_processor import NoiseTestDataProcessor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class NoiseTestDataset(Dataset):
    """Noise test dataset"""
    
    def __init__(self, data_list):
        """
        Args:
            data_list: List of processed folder samples.
        """
        self.data_list = data_list
        
        # Preprocessing pipeline for MobileViT input
        self.mobilevit_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        # Count total frames (across all folders)
        total = 0
        for data in self.data_list:
            total += data['images_ch1'].shape[0]
        return total
    
    def __getitem__(self, idx):
        # Locate the concrete folder and frame index
        current_idx = idx
        for data in self.data_list:
            T = data['images_ch1'].shape[0]
            if current_idx < T:
                # Found the matched folder/frame
                frame_idx = current_idx
                
                # Extract individual channel images
                img_ch1 = data['images_ch1'][frame_idx]  # (H, W)
                img_ch2 = data['images_ch2'][frame_idx]
                img_ch3 = data['images_ch3'][frame_idx]
                
                # Normalize to [0, 1]
                img_ch1 = np.clip(img_ch1 / 255.0, 0, 1)
                img_ch2 = np.clip(img_ch2 / 255.0, 0, 1)
                img_ch3 = np.clip(img_ch3 / 255.0, 0, 1)
                
                # Convert to tensor format (C, H, W)
                img_ch1_tensor = torch.from_numpy(img_ch1).unsqueeze(0).float()
                img_ch2_tensor = torch.from_numpy(img_ch2).unsqueeze(0).float()
                img_ch3_tensor = torch.from_numpy(img_ch3).unsqueeze(0).float()
                
                # Standardize inputs
                img_ch1_tensor = (img_ch1_tensor - 0.5) / 0.5
                img_ch2_tensor = (img_ch2_tensor - 0.5) / 0.5
                img_ch3_tensor = (img_ch3_tensor - 0.5) / 0.5
                
                # Build a three-channel image for MobileViT
                img_3channel = np.stack([img_ch1, img_ch2, img_ch3], axis=2)  # (H, W, 3)
                img_3channel_pil = Image.fromarray((img_3channel * 255).astype(np.uint8))
                img_3channel_tensor = self.mobilevit_transform(img_3channel_pil)
                
                # Compute differential displacement: disp[i] - disp[i-1], first frame is NaN
                if frame_idx >= 1:
                    displacement_gt = float(data['displacements'][frame_idx] - data['displacements'][frame_idx - 1])
                else:
                    displacement_gt = float('nan')
                
                # Keep the image index for result alignment
                image_index = data.get('image_indices', np.arange(T))[frame_idx]
                
                # Record folder path for reporting
                folder_path = data['folder_path']
                
                return (img_ch1_tensor, img_ch2_tensor, img_ch3_tensor, img_3channel_tensor,
                       displacement_gt, int(image_index), folder_path)
            
            current_idx -= T
        
        raise IndexError(f"Index {idx} out of range")


def collate_fn(batch):
    img_ch1, img_ch2, img_ch3, img_3ch, disp_gt, img_indices, folder_paths = zip(*batch)

    img_ch1 = torch.stack(img_ch1, dim=0)
    img_ch2 = torch.stack(img_ch2, dim=0)
    img_ch3 = torch.stack(img_ch3, dim=0)
    img_3ch = torch.stack(img_3ch, dim=0)

    disp_gt = np.array(disp_gt, dtype=np.float32)

    return img_ch1, img_ch2, img_ch3, img_3ch, disp_gt, list(img_indices), list(folder_paths)


class NoiseTestEvaluator:
    """Noise test evaluator"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(args.output_dir, f'noise_test_{timestamp}')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize data processor
        self.data_processor = NoiseTestDataProcessor(args.test_data_root)
        
        # Load model weights
        self.model = self._load_model()
        
        print(f"\n{'='*60}")
        print("Noise test evaluator initialized")
        print(f"Device: {self.device}")
        print(f"Model path: {args.model_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")
    
    def _load_model(self):
        """Load the pretrained MFN model"""
        print(f"[Model] Loading pretrained weights: {self.args.model_path}")
        
        # Instantiate model
        model = create_mfn_model(
            img_h=self.args.img_h,
            img_w=self.args.img_w
        )
        
        # Load weights (partial load: displacement-related heads only)
        checkpoint = torch.load(self.args.model_path, map_location=self.device)
        
        # Pull pretrained state dict
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        
        # Filter out mismatched/unwanted weights (e.g., order head)
        # Keep only image feature extractors and displacement head weights
        filtered_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                # Ensure tensor shapes match
                if v.shape == model_dict[k].shape:
                    filtered_dict[k] = v
                else:
                    print(f"[Warning] Skip weight {k}: shape mismatch (checkpoint: {v.shape}, model: {model_dict[k].shape})")
            else:
                print(f"[Warning] Skip weight {k}: not present in model")
        
        # Update model state
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        
        print(f"[Model] Loaded {len(filtered_dict)}/{len(pretrained_dict)} pretrained tensors")
        
        model = model.to(self.device)
        model.eval()
        
        print("[Model] Load complete")
        return model
    
    def evaluate_noise_level(self, noise_level):
        """Evaluate a single noise level."""
        print(f"\n{'='*60}")
        print(f"Evaluating noise level: {noise_level:.2f}")
        print(f"{'='*60}")

        # Gather all folders for the requested noise level
        folders = self.data_processor.get_all_data_folders_by_noise_level(noise_level)

        if not folders:
            print(f"[Warning] No valid folders for noise level {noise_level:.2f}")
            return None

        print(f"[Data] Found {len(folders)} folders")

        # Process folders into memory sequences
        data_list = []
        total_images = 0

        for folder_path in folders:
            data = self.data_processor.process_single_folder(folder_path)
            if data is not None:
                data_list.append(data)
                total_images += data['images_ch1'].shape[0]

        print(f"[Data] Processed {len(data_list)} folders containing {total_images} frames")

        if not data_list:
            print(f"[Warning] Noise level {noise_level:.2f} produced no valid data")
            return None

        # Build dataset and loader
        dataset = NoiseTestDataset(data_list)
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.args.num_workers,
            pin_memory=True
        )

        # Inference loop
        all_predictions = []
        all_ground_truths = []
        all_image_indices = []
        all_folder_paths = []

        total_inference_time = 0.0
        num_inference_batches = 0

        with torch.no_grad():
            for batch_data in dataloader:
                img_ch1, img_ch2, img_ch3, img_3ch, disp_gt, img_indices, folder_paths = batch_data

                img_ch1 = img_ch1.to(self.device)
                img_ch2 = img_ch2.to(self.device)
                img_ch3 = img_ch3.to(self.device)
                img_3ch = img_3ch.to(self.device)

                # Measure pure inference time (exclude data processing)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()

                predictions = self.model(img_ch1, img_ch2, img_ch3, img_3ch, mode='pretrain')

                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()

                total_inference_time += (end_time - start_time)
                num_inference_batches += 1

                # Collect outputs
                all_predictions.append(predictions.cpu().numpy())
                all_ground_truths.append(disp_gt)
                all_image_indices.extend(img_indices)
                all_folder_paths.extend(folder_paths)

        # Consolidate arrays
        all_predictions = np.concatenate(all_predictions)
        all_ground_truths = np.concatenate(all_ground_truths)

        # Filter out NaN entries (first frame lacks differential label)
        valid_mask = ~np.isnan(all_ground_truths)
        if valid_mask.sum() == 0:
            print("[Warning] No valid differential labels at this noise level; metrics skipped")
            return None

        preds_valid = all_predictions[valid_mask]
        gts_valid = all_ground_truths[valid_mask]

        # Global metrics over valid samples
        squared_errors = (preds_valid - gts_valid) ** 2
        overall_mse = np.mean(squared_errors)
        overall_rmse = np.sqrt(overall_mse)

        # Per-sample errors
        rmse_per_sample = np.sqrt(squared_errors)
        errors = np.abs(preds_valid - gts_valid)

        mean_rmse_per_sample = np.mean(rmse_per_sample)
        max_rmse = np.max(rmse_per_sample)
        min_rmse = np.min(rmse_per_sample)
        std_rmse = np.std(rmse_per_sample)
        mae = np.mean(errors)

        valid_count = preds_valid.shape[0]
        avg_inference_time_per_image = total_inference_time / valid_count if valid_count > 0 else float('nan')
        images_per_second = valid_count / total_inference_time if total_inference_time > 0 else float('nan')

        # Log results
        print(f"\n[Result] Noise level {noise_level:.2f}:")
        print("  === Global error metrics ===")
        print(f"  - MSE:  {overall_mse:.4f} nm²")
        print(f"  - RMSE: {overall_rmse:.4f} nm")
        print(f"  - MAE:  {mae:.4f} nm")
        print("\n  === Sample statistics ===")
        print(f"  - Mean RMSE: {mean_rmse_per_sample:.4f} nm")
        print(f"  - Max RMSE:  {max_rmse:.4f} nm")
        print(f"  - Min RMSE:  {min_rmse:.4f} nm")
        print(f"  - Std Dev:   {std_rmse:.4f} nm")
        print("\n  === Inference performance ===")
        print(f"  - Throughput: {images_per_second:.2f} images/s")
        print(f"  - Avg latency: {avg_inference_time_per_image*1000:.2f} ms/image")
        print(f"  - Valid frames: {valid_count}")
        print(f"  - Total inference time: {total_inference_time:.2f} s")

        # Persist per-sample records
        valid_folder_paths = np.array(all_folder_paths)[valid_mask]
        valid_image_indices = np.array(all_image_indices)[valid_mask]
        errors_valid = np.abs(preds_valid - gts_valid)
        rmse_per_sample_valid = np.sqrt((preds_valid - gts_valid) ** 2)

        results_df = pd.DataFrame({
            'Folder_Path': valid_folder_paths,
            'Image_Index': valid_image_indices,
            'Ground_Truth_nm': gts_valid,
            'Prediction_nm': preds_valid,
            'Absolute_Error_nm': errors_valid,
            'Squared_Error_nm2': (preds_valid - gts_valid) ** 2,
            'Sample_RMSE_nm': rmse_per_sample_valid
        })

        csv_path = os.path.join(self.output_dir, f'test_noise_{noise_level:.2f}_detailed_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"\n  - Detailed results saved to: {csv_path}")

        # Return aggregated statistics
        results = {
            'noise_level': noise_level,
            'overall_mse': float(overall_mse),
            'overall_rmse': float(overall_rmse),
            'mae': float(mae),
            'mean_rmse_per_sample': float(mean_rmse_per_sample),
            'max_rmse': float(max_rmse),
            'min_rmse': float(min_rmse),
            'std_rmse': float(std_rmse),
            'total_images': int(valid_count),
            'total_inference_time_sec': float(total_inference_time),
            'avg_inference_time_ms': float(avg_inference_time_per_image * 1000),
            'images_per_second': float(images_per_second),
            'num_folders': len(folders)
        }

        return results
    
    def run_evaluation(self):
        """Execute the full evaluation pipeline."""
        print(f"\n{'='*60}")
        print("Starting noise evaluation")
        print(f"{'='*60}\n")

        # Discover available noise levels
        noise_levels = self.data_processor.get_noise_levels()

        if not noise_levels:
            print("[Error] No noise levels found")
            return

        print(f"[Data] Located {len(noise_levels)} noise levels: {noise_levels}")

        # Evaluate each noise level
        all_results = []

        for noise_level in noise_levels:
            result = self.evaluate_noise_level(noise_level)
            if result is not None:
                all_results.append(result)

        # Persist overall results
        if all_results:
            summary_df = pd.DataFrame(all_results)
            summary_csv_path = os.path.join(self.output_dir, 'test_noise_summary.csv')
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"\n[Summary] CSV saved to: {summary_csv_path}")

            summary_json_path = os.path.join(self.output_dir, 'test_noise_summary.json')
            summary_dict = {
                'model_path': self.args.model_path,
                'test_data_root': self.args.test_data_root,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'results': all_results
            }

            with open(summary_json_path, 'w') as f:
                json.dump(summary_dict, f, indent=2)

            print(f"\n[Summary] JSON saved to: {summary_json_path}")

            print(f"\n{'='*60}")
            print("Noise test summary")
            print(f"{'='*60}")
            print(summary_df.to_string(index=False))
            print(f"{'='*60}\n")

        print(f"Evaluation complete! Results stored in: {self.output_dir}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Noise evaluation script')
    
    # Paths
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the pretrained model checkpoint')
    parser.add_argument('--test_data_root', type=str, 
                       default='/public_new/work_space/fengjiali/MKwithML/Noise_test',
                       help='Root directory containing noise-test data')
    parser.add_argument('--output_dir', type=str, 
                       default='./noise_test_results',
                       help='Directory for evaluation outputs')
    
    # Model dimensions
    parser.add_argument('--img_h', type=int, default=1024,
                       help='Image height')
    parser.add_argument('--img_w', type=int, default=1280,
                       help='Image width')
    
    # Evaluation settings
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    return parser.parse_args()


def main():
    """Program entry point."""
    args = parse_args()
    
    # Create evaluator
    evaluator = NoiseTestEvaluator(args)
    
    # Execute evaluation
    evaluator.run_evaluation()


if __name__ == '__main__':
    main()

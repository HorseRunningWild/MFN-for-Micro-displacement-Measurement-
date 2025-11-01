## Introduction to Core Files
---
1. `data_processor.py`:
   1. Scans the raw data directory to generate structured samples for training (images, numerical features, displacement & order labels), and provides the `MFNDataset` class and a `collate` function. *Note: For calculating the order, we use a laser wavelength of $\lambda = 633.017$nm. If you use a laser with a different wavelength, this value needs to be changed.*
   2. Processes inputs by performing operations as described in the paper, such as Fast Fourier Transform (FFT), numerical feature extraction, and sequence padding for LSTM.
   3. For labels: Applies differencing to the displacement data (in our dataset, the displacement for each image is considered absolute, not relative to the previous image). It also masks the first data point to exclude it from the MSE loss calculation.

2. `model_fenition.py`:
   1. **Image Branch**: A combination of a CNN branch (for 3 single-channel inputs) and a MobileViT branch (for the 3-channel merged input).
   2. **LSTM Branch**: Specifically used during the fine-tuning phase to capture the temporal information of noise.
   3. **Dual-Head Output**:
      - **Displacement Head**: Regresses displacement within the range of $[0, \frac{\lambda}{2}]$.
      - **Order Head**: Classifies the order into one of five categories: {-2, -1, 0, 1, 2}.
      - **Note**: To predict orders where $m > 2$, you will need to manually modify the possible range of $m$ and map the order values to a set of integer class labels, e.g., $\{0,1,\ldots,N\}$.

3. `train_batched.py`
   1. Manages batched training for large datasets by processing data in "batches of folders". It supports both pretraining and fine-tuning workflows, along with checkpoint resumption and inter-batch model accumulation.
   2. Selects different data processors and loss/logging formats based on the specified `mode` (e.g., `pretrain` uses only MSE loss, while `finetune` uses a combination of MSE + Cross-Entropy + optional orthogonal loss).
   3. Besides `Finetune` and `Pretrain`, this script also defines a `Train_again` method. This is useful for scenarios where the initial pretraining performance is unsatisfactory, or when you want to further pretrain a model (e.g., our provided one) on your own system's simulated noise. You can specify a model path to initiate another pretraining phase.
  
4. `train_variable_data_size.py`:Through this document, you can study how much data is needed to achieve good fine-tuning results. By default, we divide the fine-tuning dataset into 5 parts and successively use 1 to 5 parts of the dataset for fine-tuning, and then evaluate on the test set (the amount of test set data used can also be specified). You need to specify your model path, fine-tuning dataset path, and test set path in the corresponding `variable_data_size.sh` file.

5. `test_noise.py`
    1. Calculate the average RMSE, maximum RMSE, minimum RMSE, and standard deviation for each noise level.
    2. Calculate the inference speed for each noise level (pure inference time, excluding data processing).
    3. Please specify your dataset in the corresponding `sbatch_noise_test.sh` file

6. `sbatch_batched_pretrain.sh`: An example script for submitting jobs to a server (using SLURM's `sbatch` command). It defines nearly all the necessary hyperparameters, including training mode, dataset path, learning rate, and more. To perform pretraining, fine-tuning, or re-pretraining with our model, you typically only need to modify the parameters within this script.


## Data Preparation
---

### Data Format

The expected data format for pretraining is as follows, stored in an `.xlsx` file:

| Image Index | m Value | Displacement (nm) |
|:-----------:|:-------:|:-----------------:|
|     001     |   0.26  |       82.40       |
|     002     |   0.20  |       61.71       |
|     003     |   0.13  |       41.67       |
|     004     |   0.07  |       20.82       |

Similarly, the dataset for fine-tuning should also be in an `.xlsx` file with the following format:

| Image Index | d(nm)  |   Δm   |
|:-----------:|:------:|:------:|
|      0      | 1,423  |  -1.41 |
|      1      |  889   |  -1.69 |
|      2      |  547   |  -1.08 |
|      3      |  452   |  -0.30 |
|      4      |  412   |  -0.13 |

### Data Structure
We expect the file structure for storing data to be as follows：

Dataset_simulate_First_Time/

├─ begin_0000_end_0082/

│  └─ down_begin_0000_end_0082/

│     └─ image_001.npy

│     └─ displacement.xlsx

If you wish to use our scripts directly, please ensure your data format is consistent with the tables above.

   

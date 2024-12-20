# CLIP+ResLT Framework

## Setup Instructions

### Prerequisites
- Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.
- Install necessary libraries as outlined in `requirements.txt`.

### Installation Steps for Dassl Repo

1. **Clone the Dassl Repository**  
   Clone the Dassl PyTorch repository within the CLIP_ResLT folder:
   ```bash
   git clone https://github.com/KaiyangZhou/Dassl.pytorch.git ./CLIP_ResLT/Dassl.pytorch
   ```

2. **Add Dassl to Conda Environment**  
   Navigate to the Dassl repository and install it:
   ```bash
   cd CLIP_ResLT/Dassl.pytorch
   pip install -e .
   ```

3. **Prepare the Dataset**  
   - Modify `datasets/herbarium.py` as needed.
   - Update the `read_dataset` function to ensure correct data loading.
   - Refer to class variable comments for required fields.

4. **Reproduce the Error**  
   - Go to the scripts directory:
     ```bash
     cd CLIP_ResLT/scripts/coop
     ```
   - Modify `clip_reslt.sh` with the necessary parameters (i.e. DATA, TRAIN_META and TEST_META).
   - Recreate the error by running the script using the mentioned command to run the script in `clip_reslt.sh`. 

### Notes
- Ensure the directory structure and dataset paths are correctly specified.
- If any environment variables or config files are missing, create them accordingly.

## Troubleshooting Dassl Pytorch Library
- Refer to the Dassl PyTorch [documentation](https://github.com/KaiyangZhou/Dassl.pytorch) for detailed installation help.


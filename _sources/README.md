You can find a Google Drive folder containing some zipped initial data for the plots [here](https://drive.google.com/drive/folders/1pX7OuSbxqaFi828oOhPi67xZalIBRfcU?usp=sharing). To recreate `analysis.ipynb`, which contains both the plots and some discussion on them, first built the conda environment described in `environment.yaml` in the root directory
```bash
conda env create -f environment.yaml
```
Then download `data.zip` from the linked drive and unzip to `analysis/data`. Then activate the conda environment and run the notebook from the `analysis` directory.

```bash
conda activate o2
cd analysis
jupyter notebook analysis.ipynb
```

The code needs some cleaning and I'm only about 2/3 of the way through, but it should be enough to get the discussion going.
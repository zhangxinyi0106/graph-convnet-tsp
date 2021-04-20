# CZ4071 Project 2
Group 8: Hu Wenqi, Xia Chengguang, Zhang Xinyi, Zhou Hongyu

Paper 22: [**An Efficient Graph Convolutional Network Technique for the Travelling Salesman Problem**](https://arxiv.org/abs/1906.01227)
 by Chaitanya K. Joshi, Thomas Laurent and Xavier Bresson.

Official Codes: [graph-convnet-tsp](https://github.com/chaitjo/graph-convnet-tsp)

This repository is largely based on the official codes, but contains extra bug fixing and supports for PyTorch 1.7.1. More Details can be found in the report.
The users are recommended to read through the `README.md` of the official codes first.

## Dependencies
To use this repository, anaconda is needed.

```
conda env create -f environment.yml
conda activate gcn-tsp-env
```

In case of any errors occurred during the above process, you can always manually install the following packages:

```
conda install pytorch=0.4.1 cuda90 -c pytorch
conda install numpy==1.15.4 scipy==1.1.0 matplotlib==3.0.2 seaborn==0.9.0 pandas==0.24.2 networkx==2.2 scikit-learn==0.20.2 tensorflow-gpu==1.12.0 tensorboard==1.12.0 Cython
pip3 install tensorboardx==1.5 fastprogress==0.1.18
conda install -c conda-forge jupyterlab
```
If error message show that some particular packages cannot be found, it may be because of the platform specific problem. Please kindly
delete the corresponding package requirement from the `environment.yml`. Note that this repository supports a higher version PyTorch (1.7.1) or CUDA (10.2).
You may have them installed instead (with other dependencies updated as well) depending on your needs.

## Datasets
Download TSP datasets from [this link](https://drive.google.com/open?id=1-5W-S5e7CKsJ9uY9uVXIyxgbcZZNYBrp): 
Extract the `.tar.gz` file and place each `.txt` file in the `/data` directory. (TSP10, TSP20, TSP30, TSP50 and TSP100 are provided) 

If you wish to generate your own data:

```
cd data
git clone https://github.com/jvkersch/pyconcorde
cd pyconcorde
pip install -e .
cd ..

python generate_tsp_concorde.py --num_samples <num-sample> --num_nodes <num-nodes>
```
New dataset will be stored under `data` with name `tsp<num-nodes>_concorde_new.txt`.

For TSP50 and TSP100, the 1M training set needs to be split into 10K validation samples and 990K training samples.
You will need to run `split_train_val.py` script to do it.

```
cd data
python split_train_val.py --num_nodes <num-nodes>
```

You can create or modify existing `.json` files under the `configs` directory to use different dataset.

## Usage

#### Running in Notebook 
Launch Jupyter Lab and execute/modify `main.ipynb` cell-by-cell in Notebook Mode. Markdown titles in the notebook explain themselves.
```
jupyter lab
```

If you would like to keep the model training while testing it, you may use `evaluate.ipynb`, where the best model saved so far will be evaluated.

You need to manually change the `config_path` variable to pointing to a configuration file:
```
if notebook_mode==False:
    ...
else:
    config_path = <specify_your_config_file_here>
```

#### Running in Script Mode
Set `notebook_mode = False` and `viz_mode = False` in the first cell of `main.ipynb` or `evaluate.ipynb`.
Then convert the notebook from `.ipynb` to `.py` and run the script (pass path of config file as arguement):
```
jupyter nbconvert --to python <name>.ipynb 
python <name>.py --config <path-to-config.json>
```
**IMPORTANT:** If you forgot to change `notebook_mode = False` before converting, `--config` will be ignored. Please
keep an eye on the printed log to verify that you are running the right experiment.

## Other Notes
GPUs are strongly recommended to be used. It may take days for TSP100 model to converge even on multiple NVIDIA 2080Ti. You are recommended to
try out TSP10 before proceed to large datasets, in case of unexpected errors.

`torch==1.7.1` were tested by us on Ubuntu 16.04. For Windows10 platform, we follow the official dependencies. `environment.yml` is generated via Windows10 platform.
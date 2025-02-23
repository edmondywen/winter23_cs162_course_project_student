# UCLA CS162 Course Project Guideline

# Table of Contents

1. [Installations](#installations)
2. [Hardware Setups](#hardware)
3. [Code Executions](#executions)
4. [Datasets](#datasets)
5. [Training](#training)
   * [Visualizing Training](#tb)


# Getting Started and Installations <a name="installations"></a>

Please make sure everything is under **Python3**.

We recommend two ways of getting started to setup the necessary environment:
1. Using conda ([miniconda3](https://docs.conda.io/en/latest/miniconda.html)) (preferred).
2. Using python virtual [environment](https://docs.python.org/3/library/venv.html).


## Using Conda

Goto the above link to install the **miniconda3** corresponding to your OS.

You can use the following commands for various conda operations:

```bash
# cs162 is the name of the conda environment, you can name it anything you like.
conda create -n cs162 python==3.8

# You can list all the conda envs using the following command.
conda info --envs

# Activate the conda env.
conda activate cs162

# Initialize conda.
conda init

# Install the pip in this conda environment.
conda install pip

# And use the following to deactivate the conda environment
# if you're done with some jobs and wish to log out.
conda deactivate
```

Once you create the above environment, install all the required packages in your environment simply by:

```bash
conda activate cs162

pip install --upgrade pip

# Install PyTorch
pip install torch==1.10.1

# The below command will bulk install everything needed.
pip install -r requirements.txt
```


## Using the Virtual Environment

This is a nice [guide](https://realpython.com/python-virtual-environments-a-primer/) about setting up virtual environment, we recommend doing the followings:

```bash
python3 -m venv where_ever/you/want/cs162

source where_ever/you/want/cs162/bin/activate

# And use the following to deactivate within a virtualenv.
deactivate
```

The rest of installing the packages should remain the same as in using conda.


# Basic Hardware Guidelines <a name="hardware"></a>

For using cloud services, please refer to the slides/recording in the first TA section.
This readme will only contain more strictly project-related instructions.

## Using GPUs

When your OS features any GPUs, please only access to one of them as our task is not that heavy, and hence does not need to waste computational resources using multi-GPUs in general (but feel free to do so if you'd like). Below are commands that you can use to control GPUs while running your processes (run them before running your main script).

```bash
# Check your GPU status, this will print all the essential information about all the GPUs.
nvidia-smi

# Indicating the GPU ID 0.
export CUDA_VISIBLE_DEVICES="0"

# OR, GPU ID 2.
export CUDA_VISIBLE_DEVICES="2"

# For multi-gpu settings, suppose we would like to use 0, 1, 2 (3 GPUs).
export CUDA_VISIBLE_DEVICES="0,1,2"

# For cpu-only, and hence `no_cuda`.
export CUDA_VISIBLE_DEVICES=""
```


# Code Executions <a name="executions"></a>

1. Most of the code in the provided starter code repository should be executed using the [module option](https://stackoverflow.com/questions/22241420/execution-of-python-code-with-m-option-or-not).
   For example, if you would like to execute the code `data_processing/dummy_data.py` , do:

```bash
python3 -m data_processing.dummy_data
```

​		Notice that the `.py ` in `dummy_data.py` is gone

2. Training Arguments: 

   Before executing any training-related codes, make sure to skim through `trainers/args.py` to learn
   what arguments are used in the code. We will walk through some important ones in the **training** section in `README.md`.

3. For basic execution demo, after finishing up all the required `TODO`s in `trainers/*.py`,  
   execute the following command:

```bash
sh scripts/train_dummy.sh
```
And immediately you should see training and evaluation loops being successfully executed.

NOTE: The testing scripts are not exhaustive, but should be used as reference. Testing script working correctly doesn't ensure your code is 100% correct. There could still be some errors. 


# Datasets <a name="datasets"></a>

All the required `TODO` blocks are in the codes under `data_processing` folder.  
Please refer to the README under `data_processing` folder for data schema and execution examples.

NOTE: The testing scripts are not exhaustive, but should be used as reference. Testing script working correctly doesn't ensure your code is 100% correct. There could still be some errors. 


# Training <a name="training"></a>

Please refer to the README under `trainers` folder for details of `TODO`s.

We have prepared an example script for you to use:

```bash
# Script for finetuning a model on Com2Sense dataset.
sh scripts/train_com2sense.sh
```

We also provide some scripts that may help you further improve the model performance, the details of which can be found in the `scripts` directory.

 In any training script, if you comment out (or remove) the `--do_train` argument, then the script will only execute the testing.  
In this case, please ensure the `--iters_to_eval` is properly set to the checkpoint(s) you would like to evaluate.  

Other most essential arguments to be aware of and you **SHOULD TRY TO TUNE** are:

* `TASK_NAME` is to be consistent with `data_processing/__init__.py`.
* `DATA_DIR` to be made sure correspond to a proper dataset folder.
* `MODEL_TYPE` should be one of the [model card](https://huggingface.co/models), e.g. `bert-base-cased` 
* `model_name_or_path` can be same as `MODEL_TYPE` or a saved directory which contains the checkpoint, e.g. `outputs/pretrain/ckpts/checkpoint-100` for restoring or finetuning a trained checkpoint.
   * Remember that you **CAN ONLY** use one of the following models for training on both Sem-Eval and Com2Sense: `{bert, deberta, roberta}`.
* `tokenizer_name` and `config_name` can be both not specified (hence refer to `model_name_or_path`) or same as `MODEL_TYPE`.
* `per_gpu_train_batch_size` and `per_gpu_eval_batch_size` can be used to tune training/evaluation batch size.
* `learning_rate` tunes the initial learning rate.
* `num_train_epochs` maximum number of epochs to train the model.
* `max_seq_length` the maximum sequence length of inputs to the model.
* `output_dir` where to save your outputs and checkpoints.
* `save_steps` denotes per how many steps we save the models.
* `logging_steps` denotes per how many steps we evaluate the models during training.
* `max_eval_steps` maximum evaluation steps for an evluation split of a dataset.
* `eval_split` the split to be evaluated on, during training it should be `dev` and it should be `test` during testing.
* `iters_to_eval` the iterations of the saved checkpoints to be evaluated on.
  * If you implement saving the best functionality, it can also be `best` instead of a number.

## Visualizing Your Training <a name="tb"></a>

It is often important to visualize your training curves and other essential information during training for troubleshooting problems or ensuring your training is stable (e.g. observing if your training is over/under-fitting).
People often use [TensorBoard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) (the PyTorch compatible version) for this purpose.  
Please check the introduction in the link to get familiar with this very useful tool.

Each execution of your codes will automatically create a subfolder under the folder `runs`, e.g. `runs/Jan01_06-03-20_pluslab-a100_ckpts_dummy` can be a subfolder containing tensorboard events of one of your execution.  
Executing the following command a tensorboard-in-browser will be rendered:

```bash
tensorboard --logdir=the/tensorboard/event/dir

# For example:
tensorboard --logdir=runs/Jan01_06-03-20_pluslab-a100_ckpts_dummy
```
You should see the followings as an output:
```bash
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.7.0 at http://localhost:6006/ (Press CTRL+C to quit)
```
Since the `localhost` is on the server, unless you can directly access to the GUI on the server, you may find this **[SSH Tunneling](https://coderwall.com/p/ohk6cg/remote-access-to-ipython-notebooks-via-ssh)** handy.  
For example, if you execute the following command **on your local terminal**:
```bash
ssh -N -f -L localhost:6006:localhost:6006 your_id@the_server_name
```
Then, open up a browser on your machine and in the search bar enter `http://localhost:6006/` you will see the rendered tensorboard shown up!

# FSDP Experiments

## Setup Env

I've used Conda for all my experiments and it has served me well. You may choose to use other environment managers such as Mamba, Poetry etc. and your Mileage may vary but any mature environment manager should do the job.

For Conda Users - 

- The python package details are present in `environment.yml`. The exact build details alongside python packages are present in `environment_with_build_info.yml`
- To use the env file run - `conda env create -f YAML_FILE_PATH`

## Running Instructions 

Since I am using Accelerate for DDP and FSDP, it is possible but hard to run it from the notebooks. So for the DDP and FSDP variants I have added `.py` files which can be run using `accelerate launch` commands.

For FSDP - `accelerate launch --config_file fsdp_config.yaml train_model_fsdp.py`
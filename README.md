Paper Link: https://www.sciencedirect.com/science/article/abs/pii/S0957417423006383

# Koopman-Mixup (K-mixup)
Environment setup for K-mixup

1. Create new conda environment
```python
conda create -n ENV_NAME python=3.6.10
```
2. Install d3rlpy
```python
pip install pillow
pip install PyYAML
pip install d3rlpy
```
3. Install dm_control
```python
pip install dm_control
```

4. Install d4rl
```python
cd d4rl
pip install -e .
```

5. Install Flow
```python
git clone https://github.com/rail-berkeley/d4rl.git
cd flow
python setup.py develop
scripts/setup_sumo_ubuntu18.04.sh
```

6. Install Mujoco

- Download Mujoco 2.0.2 binary file in https://roboti.us/download.html
- Download activation key in https://roboti.us/license.html
- Extract the binary file at /home/{USER_NAME}/.mujoco/mujoco200
- Copy the activation key at /home/{USER_NAME}/.mujoco/mujoco200 and /home/{USER_NAME}/.mujoco/mujoco200/bin

```python
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
conda install -c anaconda patchelf
pip install -U ‘mujoco-py<2.1,≥2.0’
```
- Copy the following to the ~/.bashrc
```python
export MUJOCO_GL=osmesa
export MJLIB_PATH=$HOME/.mujoco/mujoco200/bin/libmujoco200.so
export MJKEY_PATH=$HOME/.mujoco/mujoco200/mjkey.txt
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200/bin
export MUJOCO_PY_MJPRO_PATH=$HOME/.mujoco/mujoco200/
export MUJOCO_PY_MJKEY_PATH=$HOME/.mujoco/mujoco200/mjkey.txt
```
7. Install Tensorflow == 1.15.2
```python
pip install tensorflow==1.15.2
```

8. Install Torch >= 1.7
- For RTX3090 
```python
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

9. Install scikit-learn 

```python
conda install -c conda-forge scikit-learn
```

10. Install progressbar
```python
conda install -c conda-forge progressbar
```
***
- Implementation of DVK model is forked from the official github page. (Original Implementation https://github.com/sisl/variational_koopman)
- Original D4RL code is modified to include two novel datasets: reacher-medium-v0, swimmer-medium-v0 (Original Implementation: https://github.com/rail-berkeley/d4rl) 

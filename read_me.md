## Create new conda env
conda create --name mixup python=3.6.10

## Install DVK
git clone https://github.com/sisl/variational_koopman.git

## Install d3rlpy (An offline deep RL library)
pip install d3rlpy -> for recent version, maybe install with soruce code like below d4rl

pip install dm_control
pip install git+git://github.com/aravindr93/mjrl@master#egg=mjrl


git clone https://github.com/rail-berkeley/d4rl.git
cd d4rl
pip install -e .

--> d4rl 설치할 때, dm_control 부분 setup.py 에서 제외하고 따로 설치함 (git으로 설치하니까 오류나서..)

Requires Mujoco, Flow, Carla (go to https://github.com/rail-berkeley/d4rl)

## Mujoco install
pip3 install -U 'mujoco-py<2.2,>=2.1'
-> 이거 말고 예전거로 설치해야 하는듯?  (2.0 버전 직접 다운로드)
When AttributeError : mj_certQuestion 발생시 -> sudo apt-get install libglfw3 libglew2.0


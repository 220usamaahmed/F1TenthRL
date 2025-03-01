# Humanoid Robotics Lab (F1Tenth)

## Dependencies
openai gym > f1tenth_gym > f1tenth_gym_ros > ws24_racing

### f1tenth_gym

#### Running this repo
```bash
# Use this fork (exposes lidar params and env is compatable with SB3)
git clone git@github.com:220usamaahmed/f1tenth_gym.git

conda create --name f1tenth python=3.8
conda activate f1tenth

# Change 0.19.0 -> 0.26.2 in setup.py

cd f1tenth_gym
pip install -e .

pip install pyglet==1.5.20

pip install pip==19.2.3
pip install setuptools==65.5.0

pip install gym==0.19.0 # Ignore any errors related to opencv

cd examples
python3 waypoint_follow.py
```

#### Custom maps
[Build custom maps](https://f1tenth-gym.readthedocs.io/en/latest/customized_usage.html)

#### Running with ROS

#### Final Set of Dependencies
```
------------------ ------- ------------------------------------------------------------
cloudpickle        1.6.0
f110-gym           0.2.1   /Users/usama/UniversityWork/HumanoidRobotics/f1tenth_gym/gym
future             1.0.0
gym                0.19.0
gym-notices        0.0.8
importlib-metadata 8.5.0
llvmlite           0.41.1
numba              0.58.1
numpy              1.22.0
pillow             10.4.0
pip                19.2.3
pyglet             1.5.20
PyOpenGL           3.1.7
PyYAML             6.0.2
scipy              1.10.1
setuptools         65.5.0
wheel              0.44.0
zipp               3.20.2
```

Install f1tenth_gym with gym 0.26.2, then downgrade pip and setuptools and install gym 0.19.
Trying to directly install 0.19 leads to setup.py failing and nothing getting installed.

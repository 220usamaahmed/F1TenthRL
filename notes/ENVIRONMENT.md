# F1Tenth environment

## Code structure

- f110_env.py 
    - F110Env
        - Gym environment
        - Takens map, car and LIDAR parameters, number of agents, integration timestep, main car idx (ego_idx) 
        - Creates Simulator 

- base_classses.py
    - Simulator
        - Handles the interactions and update of all vehicles
        - Takes car and LIDAR parameters, num_agents ...
        - Map is set from F110Env
        - RaceCar are created for all agents

    - RaceCar
        - Handles physics and LIDAR scan of single vehicle
        - Takes car and LIDAR parameters
        - Creates ScanSimulator2D

- laser_models.py
    - ScanSimulator2D
        - 2D LIDAR scan simulator
        - Takes num_beams and fov ...

- collision_models.py
- dynamic_models.py
- rendering.py

## Actions space
These are part of the car parameters passed to F110Env

- Velocity:
    - v_min = -5.0
    - v_max = 20.0
    - a_max: 9.51

- Steering
    - s_min: -0.4189
    - s_max: 0.4189
    - sv_min: -3.2
    - sv_max: 3.2

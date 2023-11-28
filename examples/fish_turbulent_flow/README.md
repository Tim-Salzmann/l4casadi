# Energy Efficient Fish Navigation in Turbulent Flow

This example shows how to design an optimization-based trajectory generator to navigate a turbulent fluid flow.
The flow is modelled by a Neural Network in PyTorch, while the optimal trajectory generator is written in CasADi.
By using L4CasADi, both can be combined and optimized jointly. When doing so, we seek to find the minimum energy
trajectory that allows a fish to navigate from the starting point to the goal. For this purpose, the fish needs to
swim up stream a river where a circular stone causes the flow to be turbulent. 

<div align="center">
  <img src="./media/trajectory_generation_vorticity.gif" alt="Energy Efficient Fish Navigation in Turbulent Flow" width="600" height="300">
  <p><i>Energy Efficient Fish Navigation in Turbulent Flow</i></p>
</div>

## Setup
Download the data for the example [here](https://drive.google.com/file/d/1_amoosdtTko61gzUIX4D6LxpuQ_Dm0ka/view?usp=sharing)
and extract it into the sub-folder `data`. The resulting folder structure should look like this:
```
fish_turbulent_flow
├── data
│   ├── CC.csv
│   ├── UALL.csv
│   ├── VALL.csv
│   ├── VORTALL.csv
│── media
│── ...
```

Make sure to install the additional requirements for this example:

```pip install -r requirements.txt```

To generate animations `ffmpeg` is required.

Linux: ```sudo apt install ffmpeg```.\
OSX:  ```brew install ffmpeg```.

## Run the Trajectory Generation
Run 

```python trajectory_generation.py```

This will use a pre-trained model to optimize for a energy efficient trajectory.

<div align="center">
  <img src="./media/trajectory_generation_velocity_field.gif" alt="Energy Efficient Fish Navigation in Velocity Field" width="600" height="300">
  <p><i>Energy Efficient Fish Navigation in Velocity Field</i></p>
</div>

## Train the PyTorch Model from Scratch
To train the model from scratch, run the following command first:

```python generate_interpolators.py```

Be patient, this might take a bit. Once the interpolators have been generated, we train the network in a supervised
learning fashion with the following script:

```python learn_turbulent_flow.py```
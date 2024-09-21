# edge_image
unclear target and background create

## Update!!!
**pose/face mask function added**

# show Edge emphasis image with illumination 

```

python life_lightning.py

```
- other scripts are coloring variation

# sample Image

![Test Image 3](sample_image.png) 


# Pose and face 3D point estimation

- calibration script
  - Generating json file which is recorded face moving
```

python carib_face.py
```
## If you want to replace model, you can check available model

```Python:model_check.py
import torch

# show available model list
available_models = torch.hub.list('intel-isl/MiDaS', force_reload=False)
print("available model:", available_models)
```



- main script
  
```

python firelight.py

```
- Press s key: you can show Mediapipe poseestimation original screen
- q key is quit script all.

# Example

![Test Image 3](3dimage.png) 


# Requirement 

```
pip install open3d
pip install 
pip install opencv-python numpy


```

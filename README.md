# Rainfall Downscaling Model

This repository contains a deep learning model for **rainfall downscaling** from coarse to high resolution.  
The corresponding paper is currently under submission.


## Key parameters
--upsampling_factor

Spatial downscaling factor between input data (low-resolution) and output data (high-resolution).

--dim_channels

Number of input/output channels.

--tau

Rainfall threshold used to define rain / no-rain.


## Acknowledgements
The basic cnn architecture and parts of the data loading pipeline are adapted from:
RolnickLab – constrained-downscaling
https://github.com/RolnickLab/constrained-downscaling

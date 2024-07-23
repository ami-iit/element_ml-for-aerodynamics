# element_ml-for-aerodynamics

## Responsible 

[Antonello Paolino](https://github.com/antonellopaolino)     |
:-------------------------:|
<img src="https://github.com/antonellopaolino.png" width="180"> |  

## Background
The work previously done in [element_automatic-cad-cfd](https://github.com/ami-iit/element_automatic-cad-cfd) and [element_cfd-simulation](https://github.com/ami-iit/element_cfd-simulation) for simulating and estimating the aerodynamic forces acting on iRonCub robot during flight, and the aerodynamic modelling and control activities carried on in [element_aerodynamics-control](https://github.com/ami-iit/element_aerodynamics-control), highlighted the necessity to adopt more complex aerodynamic models to fit the highly-nonlinear data coming from CFD simulations.
Deep Learning Neural Networks have already been successfully employed to map the robot steady state (wind velocity orientation and joint positions) to the aerodynmaic forces of the links, providing an efficient and accurate data-oriented methodology to model aerodynamics.
The final objective will be the construction of physics-informed learning algorithms capable of modelling the complex nature of the physical quantities (pressure and wall shear stress) determining the aerodynamic forces and moments acting on the robot. 

## Objectives

### Study the state-of-the-art
The first step will be the study of the literature on learning methods for similar problems to understand the most suitable ones for our specific use-case.


## Outcomes

### Select suitable learning algorithms
After the state-of-the-art analysis, we'll select the most suitable learning algorithm family to carry on the activity.

### Train learning algorithm
We'll train the selected algorithm to be able to predict the flow quantities of interest according to the robot state.

### Inject flow physics information
We'll eventually try to inject physics information of the flow in the learning method (e.g. by adding Physics-Informed terms in the training procedure). 


## Milestones 

### [Study Physics-Informed Autoencoders for real-time flow prediction]()

# Remarks
## CAD models
The CAD models in this repository have been designed using [PTC Creo](https://www.ptc.com/en/products/cad/creo). Refer to [this guide](https://github.com/loc2/loc2-commons/wiki/Setup-PTC-Creo) to configure the shared libraries (e.g. for commercial components).

## Git LFS remark
This repository exploits the Git Large File Support ([LFS][1]) to handle the binary files (e.g. PTC Creo and PDF files). To download the binary files, follow the GitHub [instructions][2].

[1]:https://git-lfs.github.com/
[2]:https://help.github.com/articles/installing-git-large-file-storage/

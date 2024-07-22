# element_ml-for-aerodynamics

## Responsible 

[Antonello Paolino](https://github.com/antonellopaolino)     |
:-------------------------:|
<img src="https://github.com/antonellopaolino.png" width="180"> |  

## Background
The work previously done in [element_automatic-cad-cfd](https://github.com/ami-iit/element_automatic-cad-cfd) and [element_cfd-simulation](https://github.com/ami-iit/element_cfd-simulation) for 

Francesco Nori, Daniele Pucci and Silvio Traversaro have analized and simulated the Momentum Control of an Underactuated Flying Humanoid Robot in order to study the flight dynamics of iCub robot in horizontal and vertical motions.
In those analysis the aerodynamic forces acting on the iCub robot are neglected. In order to decide if these forces can be neglected compared to other forces it will be useful to study, in the first analysis, the aerodynamic forces acting on simple geometries.
The previous work done by Giovanni Trovato underlined the importance to establish a correct numerical method and a properly made discretization of the flow domain to have accurate results by the numerical simulations, in addition the software Ansys Fluent has been chosen to run the CFD analisys.
The final stage of this work will be the analysis of the aerodynamics of the complete iCub robot.

## Objectives

### Evaluate different turbulence models
Since the problem deals with the aerodynamics of a bluff body as the iRonCub is, it's fundamental to determine which turbulence model returns the best results in terms of forces and moments prediction accuracy and computational cost.

### Estimate aerodynamic coefficients on iRonCub robot different parts.
With CFD analysis it will be possible to estimate aerodynamic forces and moments relative to the different rotation angles of the iCub parts separated.

### Estimate aerodynamic forces and moments on the complete iRonCub robot.
The same as before can be done on the complete iRonCub robot estimating the aerodynamic forces and moments acting on it during flight.

### Validate CFD numerical results with wind tunnel experiments data
The CFD simulations performed on the iRonCub whole-body geometry in flight conditions will be validated through wind tunnel tests which will provide a quite reliable reference for the forces and moments acting on the robot and also for the pressure distribution which generates them.
## Outcomes

### Estimate aerodynamic forces
Depending on the results obtained from the simulations it will be possible to understand the weight of the aerodynamic forces acting on iRonCub compared with the dynamic forces.

### Use CFD simulations on iRonCub to build an aerodynamic model
Next step is to calculate the aerodynamic forces on the complete iRonCub verifying if it matches the sum of the single parts aerodynamic forces allowing the use of the superposition method or not (given the strong wake interations between the different parts we expect the superposition to be useless for this kind of problem). This will be cross-validated through wind tunnel experiments.

## Milestones 

### [Analyse data from wind tunnel test](https://github.com/ami-iit/element_cfd-simulation/issues/105)
After the wind tunnel tests performed @ Polimi it is necessary to perform a post-processing analysis to evaluate the experiment results (in terms of forces and moments and pressure distribution) and their consistency with the CFD simulations results.

### [Reduce the gap between CFD and wind tunnel tests results](https://github.com/ami-iit/element_cfd-simulation/issues/113)
After comparing the data between CFD and wind tunnel experiment, it will be possible to act on the simulations numerical models and parameters to reduce the gap between their results.

### [Understand the performances of Lattice-Boltzmann Method for iRonCub CFD simulations](https://github.com/ami-iit/element_cfd-simulation/issues/142)
We decided to investigate a promising frontier of CFD simulations exploting GPUs capabilities called Lattice-Boltzmann Methods; despite being already present in literature for internal or multi-phase fluids CFD simulations, in the last years the development of much more powerful GPUs has raised the interests in external aerodynamic simulations. We will compare and evaluate the performances of LBM with the more classical approach of Finite Volume Methods as implemented in ANSYS Fluent, to better understand the suitability of FluidX3D software for our scientific scopes.

# Remarks
## CAD models
The CAD models in this repository have been designed using [PTC Creo](https://www.ptc.com/en/products/cad/creo). Refer to [this guide](https://github.com/loc2/loc2-commons/wiki/Setup-PTC-Creo) to configure the shared libraries (e.g. for commercial components).

## Git LFS remark
This repository exploits the Git Large File Support ([LFS][1]) to handle the binary files (e.g. PTC Creo and PDF files). To download the binary files, follow the GitHub [instructions][2].

[1]:https://git-lfs.github.com/
[2]:https://help.github.com/articles/installing-git-large-file-storage/

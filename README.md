# Volumetric Fusion 
Final project for the [3D Scanning &amp; Spatial Learning practical course](https://www.in.tum.de/cg/teaching/winter-term-1920/3d-scanning-spatial-learning/) from the [Chair of Computer Graphics and Visualization](https://www.in.tum.de/en/cg/startseite/) at [Technische Universität München](https://www.tum.de/) in the WS 2019/20.  
The course is held by [Dr. Justus Thies](https://www.niessnerlab.org/members/justus_thies/profile.html) and our group is supervised by [Aljaž Božič](https://niessnerlab.org/members/aljaz_bozic/profile.html).

## Overview
A multi-view RGB-D Capture Setup is built using 4 [Intel® RealSense™ Depth Camera D415](https://www.intelrealsense.com/depth-camera-d415/) to perform real-time reconstructions of a moving scene.

## Steps
- Image preprocessing
  - Depth filtering
  - Background subtraction
  - Visual hull computation
- Non-rigid shape reconstruction
  - Sparse voxel grid
  - Deformation graph construction
- Non-rigid tracking
  - Projective depth ICP
  - Global sparse correspondences
  
## Data
- 4 [Intel® RealSense™ Depth Camera D415](https://www.intelrealsense.com/depth-camera-d415/)

## Literature
- [Motion2Fusion: Real-time Volumetric Performance Capture [Dou et al.]](http://library.usc.edu.ph/ACM/TOG%2036/content/papers/246-0008-dou.pdf)
- [Fusion4D: Real-time Performance Capture of Challenging Scenes [Dou et al.]](https://www.samehkhamis.com/dou-siggraph2016.pdf) 
- [DynamicFusion: Reconstruction and Tracking of Non-rigid Scenes in Real-Time [Newcombe et al.]](https://rse-lab.cs.washington.edu/papers/dynamic-fusion-cvpr-2015.pdf)

---

# Progress

This section gives a short overview over the progress of the project and about what we are doing.

## Steps
- Image preprocessing &#x2612; 
  - Depth filtering &#x2612;
  - Background subtraction &#x2612;
  - Visual hull computation &#x2612;
- Non-rigid shape reconstruction &#x2612;
  - Sparse voxel grid &#x2612;
  - Deformation graph construction &#x2612;
- Non-rigid tracking &#x2612;
  - Projective depth ICP &#x2612;
  - Global sparse correspondences &#x2612;

## Done
- C++ project setup
- [OpenGL](https://www.opengl.org/) and [GLFW](https://www.glfw.org/)
  - Get in touch with OpenGL and GLFW by doing the [OpenGL tutorial](https://learnopengl.com) to implement a visualization program for the reconstructed scene
  - Basic window rendering and vertex buffer, vertex array and vertex index setup working
  - Basic shader program added to render the scene

## Planned
- [OpenGL](https://www.opengl.org/) and [GLFW](https://www.glfw.org/)
  - Gather further knowledge over OpenGL and GLFW by doing the [OpenGL tutorial](https://learnopengl.com) 
  - Improve the visualization program to render dynamically generated scenes
- Basic Setup & Camera setup, Recordings, Visualization code
  - 17.10. – 24.10.
- Camera calibration
  - 24.10. – 21.11
- Voxel Grid implementation
- Fusion of Voxel Grids into Depth maps
  - Backprojection
- Rigid reconstruction 
- Canonical model
- Non-rigid reconstruction
- Tracking


## Ongoing tasks
- Optimization of implemented algorithms
- Migration of CPU code to GPU

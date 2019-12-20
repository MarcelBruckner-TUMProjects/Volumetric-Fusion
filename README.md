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
- [RealSense Multiple Camera WhitePaper](https://www.intel.com/content/dam/support/us/en/documents/emerging-technologies/intel-realsense-technology/RealSense_Multiple_Camera_WhitePaper.pdf)
- [Volumetric Capture](https://github.com/VCL3D/VolumetricCapture)
- [Voxel grid TSDF](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=2ahUKEwjW972elb_mAhWEs3EKHWG9BuMQFjAAegQIBRAK&url=https%3A%2F%2Fvision.in.tum.de%2F_media%2Fteaching%2Fss2015%2Fgpucourse_ss2015%2Fkinectfusion.pdf&usg=AOvVaw0IM23fQBT9kj6x-c594nz2)

## Github Repos
- [tsdf Fusion](https://github.com/andyzeng/tsdf-fusion)
- [tsdf Fusion in Python](https://github.com/andyzeng/tsdf-fusion-python)
- [Elastic Fusion](https://github.com/mp3guy/ElasticFusion)
- [ReFusion](https://github.com/PRBonn/refusion)
- [KinectFusion](https://github.com/chrdiller/KinectFusionApp)
- [ReFusion](https://github.com/PRBonn/refusion)

# Superresolution and Prediction of Ocean Sea-Surface Temperature

## Project Outline

Ocean sea surface temperature (SST) is one of the fundamental controls over our weather and climate. SST anomalies can lead to weather anomalies that cause droughts, floods, and other emergencies which strongly impact our society.

This project will explore the application of machine learning to augment the SST observed by satellite. The satellite data have two main limitations:

1. limited spatial resolution
2. no observations of the future!

Emerging techniques from deep learning have the potential to help on both counts. We will adopt the technique of "image superresolution" via generative adversarial neural networks from computer vision to the problem of SST. A training dataset of high-resolution SST images will be used to train a model which can effectively enhance the resolution of coarse-resolution satellite images. This problem is very clearly posed and should be a straightforward application of existing techniques. As such, it provides an ideal entry point into oceanography for a data science student.

The second topic involved prediction of SST. This is more challenging. In collaboration with Prof. Carl Vondrick of Computer Science, we will adopt a self-supervised learning technique used to predict video to the problem of SST. The goal is to learn how forecast the SST for days and weeks into the future given observations of the past and present state. This is a more challenging project, because it requires more domain-specific knowledge and requires a deeper understanding of machine learning algorithms.

##  Datasets

Two datasets will be used:

NASA JPL Multi-scale Ultra-high Resolution Sea Surface Temperature
https://mur.jpl.nasa.gov/
The data are distributed in netCDF format, but we will generate a cloud-native copy in Zarr format which is optimized for machine learning pipelines.

MITgcm LLC4320 SST
These are simulated SSTs which can be used as a testbed for training and prediction.
https://medium.com/pangeo/petabytes-of-ocean-data-part-1-nasa-ecco-data-portal-81e3c5e077be
These data have already been converted to a cloud-native analysis-ready format and are stored on google cloud:
https://pangeo-data.github.io/pangeo-datastore/master/ocean/llc4320.html

Both datasets will use the new cloud-native Zarr format:
https://zarr.readthedocs.io

## Specific Goals

- Success on topic 1 (SST superresolution) would mean the ability to skillfully predict a high resolution (e.g. 1km pixel size) image from a medium resolution one (e.g. 20km pixel size).
- Success on topic 2 (SST prediction) would mean the ability to forecast SST a week in advance based on the previous week of data.

In both cases, it is also important to understand which deep learning architectures are effective for this type of oceanographic application and how model design differs from classic computer vision applications.


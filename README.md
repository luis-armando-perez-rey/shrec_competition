# SHREC (SHape REtrieval Competition) Image Based 3D Scene Retrieval with Conditional Variational Autoencoders  
This repository includes the code submitted for the [SHape REtrieval Competition of 2019](http://orca.st.usm.edu/~bli/SceneIBR2019/) on image based 3D scene retrieval. The main task of the competition was to create a model capable of retrieving the most relevant 3D scenes based on an image query. Relevant scenes are such that share the same class type with the query image.
 

VGG weights pretrained on the [Places](http://places2.csail.mit.edu/) dataset were obtained from 
by [Kalliatakis G.](https://github.com/GKalliatakis/Keras-VGG16-places365). 


#### Rendering
The method works with rendered images from the sketchup models.
The renders are generated from the Ruby scripts within the rendering folder. The scripts need to be run in the SketchUp Ruby terminal. 

#### Dataset
In the dataset folder folder the data from the [SHREC competition](http://orca.st.usm.edu/~bli/SceneIBR2019/) needs to be added. The generated renders for both the train and test 3D models are placed in render_256 and test_render_256 respectively. On the other hand, the data corresponding to the query images both for training and test need to be placed at the train_photographs and test_photographs folder respectivel.   


#### Notebooks
The order for running the notebooks is as follows
1. resize_images: This notebook processes the data that has been placed in the corresponding folder to generate the resized data for training and evaluation. 
2. train_simple_cvae or train_vgg_cvae: These notebooks are used for training the corresponding models. Models are placed in the training_models folder.
3. make_submissions_simple or make_submissions_vgg: These notebooks are used for evaluating the trained models which have been placed in the trained_models folder after training. The submission files are placed in the submissions folder  

#### References

- Reference to Places dataset
 
Places: A 10 million Image Database for Scene Recognition
Zhou, B., Lapedriza, A., Khosla, A., Oliva, A., & Torralba, A.
IEEE Transactions on Pattern Analysis and Machine Intelligence
- Reference to Kalliatakis repository:

@misc{gkallia2017keras_places365,
title={Keras-VGG16-Places365},
author={Grigorios Kalliatakis},
year={2017},
publisher={GitHub},
howpublished={\url{https://github.com/GKalliatakis/Keras-VGG16-places365}},
}


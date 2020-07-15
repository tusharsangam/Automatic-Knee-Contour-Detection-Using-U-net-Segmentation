# DogBreedClassification-CNN-libtorch-Cpp
Dog breed classification by Convolution Neural Networks implemented using libtorch(Pytorch C++ library), Udacity CppND Capstone Project.
The project idea is derived from Udacity's Deep Learning Dog Breed Classification project and extended in C++.
The [original project](https://github.com/tusharsangam/deep-learning-pytorch-dog-breed-classification) was made in Pytorch and Python. Borrowing from same idealogy this project creates three types of CNNs to identify amongst 133 Dog Breeds.  
1. Pretrained VGG-16 Net to identify whether the given image is of Dog
2. A CNN architecture made from scratch(as expected this performs poor)
3. Transfer Learning Net a hybrid of VGG-16 feature extractor and custom fully connected layers for inference.

## Dependencies for Running Locally
* cmake >= 3.0
  * Linux: [click here for installation instructions](https://cmake.org/install/)

* make >= 4.1 (Linux)
  * Linux: make is installed by default on most Linux distros

* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros

* Visual Studio 2019 (Windows) 

* OpenCV >=3.0
  * OpenCV is used for reading the images and doing some primary augmentations like resize, rotate, etc, thus any latest working version of opencv can suffice but I used 4.2.0
  * The OpenCV 4.2.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.2.0)
  * Prebuilt Binaries for Windows can be found [here](https://sourceforge.net/projects/opencvlibrary/files/4.2.0/opencv-4.2.0-vc14_vc15.exe/download)
  * On Windows open the .sln file and change the include directories and link directories to point to OpenCV prebuilt binaries folder. Also include the opencv.lib static library and place .dll file into the build directory of the project or you can alternatively add the path to .dll in computer's environment variable.  

* Boost >=1.40 
	* Boost is used for directory listings since the C++ std is limited to C++14 imposed by libtorch we cannot use C++ filesystem API introduced in C++17 standards
	* Linux: `sudo apt-get install libboost-dev` might work
	* Windows: Prebuilt Windows Binaries are available [here](https://sourceforge.net/projects/boost/files/boost-binaries/)
	* To link it in the project follow similar process add include directory in project properties and add the lib directories. [Boost official installation guide](https://www.boost.org/doc/libs/1_55_0/more/getting_started/windows.html) check the section 4.1

* Libtorch >=1.4
	* Comes in Cuda and CPU only flavours
	* Instructions to download and install are available on official [Pytorch website](https://pytorch.org/cppdocs/installing.html)
	* Add the folder path from libtorch which contains .dll files to system environment
	* For Linux download [cxx11 ABI](https://download.pytorch.org/libtorch/cu92/libtorch-cxx11-abi-shared-with-deps-1.4.0%2Bcu92.zip)
	* To integrate it in windows visual studio follow this [blog post](https://medium.com/@boonboontongbuasirilai/building-pytorch-c-integration-libtorch-with-ms-visual-studio-2017-44281f9921ea)

* Original Build Machine : Windows 10, Visual Studio 2019, GTX 1080 graphic card with cuda 10.1

## Basic Build Instructions
* For Linux:
	1. Clone this repo.
	2. In root folder run : `make build`
	3. Run it: `./bin/pytorchcnn`.
	4. Some path references may have to be changed if it doesn't recognize the paths
* For Windows:
	1. Install Visual Studio
	2. Clone repo and open .sln file
	3. Change library references and build the project

## Downloading Weights and Dataset
* Make a new folder in the repo called weights and download all the pretrained weights into it
* Download [VGG-16 pretrained weights](https://drive.google.com/file/d/15WITbWa42cVBDlkY-70Tgyb8TBrYZUfs/view?usp=sharing)
* Download [VGG-16 transfer net weights](https://drive.google.com/file/d/10PVxDznNyYYGz_rGsUx4QzNzMt_OjMMQ/view?usp=sharing)
* Download [Weights for scratch model](https://drive.google.com/file/d/1ddRn7qBIkXeRlS9xJvsi0umjeSJfo4_r/view?usp=sharing)
* Place the dataset dogImages folder in the main repo directory
* Download [dogImages dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
* All the download links are also provided in [links.txt](./links.txt)

## Setting up Debug For Windows
* In visual studio the debug environment and working directory are different and these setting scan be changed in properties -> debugging 
* If visual studio debugger has to be used the .dll file path from libtorch needs to be appended to the environment variable in debug settings

## Dataset Structure
* Dataset root folder name is dogImages and it has three sub folders
	1. Train
	2. Test
	3. Valid
* Each of these subfolders have exactly 133 subfolders with corresponding dog breed names
* Each of the breed subfolders have some number of images of that particular dog breed
* For every dogimage its immediate parent folder is the label name

## Expected Behaviour
* On running the project executable it will give you a choice to choose the network between three available types of networks
* It will ask you whether you want to train, test or evaluate that particular network
* Training and testing will train and test on entire dataset thus requires more time and compute resources
* While Evaluate lets you run the network on the custom images that you pass.
* These custom images can be placed in [evaluation](./evaluation) folder
* The program will go over each image in the folder and will try to predict the dog breed (however the fully pretrained vgg-16 network will only detect the dog)

## File I/O
* a major challenge I encountred was that I was unable to import pytorch's pretrained weights into the libtorch project even after having the similar model structure
* Python's pickling isn't directly compatible with C++, however networks made in pytorch can still run in C++ but those map to the exact directory structures and are not exactly portable
* I wanted to make a network in C++ and load the pretrianed weights as done by the torchvision.models, [VGG-16 implementation of torch vision](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py)
* Inspired from [Kolkir's Json serializing and deserializing](https://github.com/Kolkir/mlcpp/blob/master/mask_rcnn_pytorch/stateloader.h) and getting to know that it was slow as json is processed in text format I had used binary file reading and writing method
* Python folder in main git repo has an extractor python script which loads the vgg-16 pretrained network and extracts all of the model's parameters in binary format
* While C++ knowing the exact shape and type details of the tensor can load exactly the required amount of bytes from the file into the C++ torch tensor
* In C++ read method a char* to chararray of lenght should be passed in order to read lenght bytes into the array
* The typesize * shape give the exact number of bytes that needs to be read while tensorobj.data_ptr interfaces the underlying allocator object of tensor.
* detailed code can be found in model.cpp load or save method
* The ordered dictionary of parameters both in C++ and Python takes care of the order in which tensors are saved and loaded so the weights can be loaded in correct order
* While saving the weights torch::save method can be used but I used the plain binary file writing and reading mechanisms provided by fstream in C++

## DataSet and Dataloader
* The Source.cpp loads model.h, dataloader.h and transforms.h
* Dataset.h and Dataloader.h takes care of dataloading and preprocessing
* The design of Dataset class is influenced from the inbuilt dataset class from the libtorch
* Initially I had extended my dogbreeddataset class to the dataset class in the libtorch but soon there was lot of confusion and possible failure points since it was my first time to use pytorch so I used my custom dataset class and dataloader class with randomsampler provided by the libtorch
* The Dataset class on intialization scans the particular dataset directory and lists all the image files in it
* Additionally the dataset takes a lambda function of preprocess transforms as constructor argument and applies that on each image data when producing batches
* Label is decided by the immediate parent's folder name, text labels are converted to int labels, like first text label is label 0, next text label is label 1 and so on.
* The dataset has three main methods, 
	1. `get(data_at_index)` reads the image and label at particular index 
	2. `get_batch(batch_indices)` calls above method for particular batch indices and stacks them and converts to a Example struct of format `{data_tensor, label_tensor, batch_idx}`
	3. `size()` returns the total dataset size i.e. the number of image paths available
* get_batch also loads 3 threads in parallel to load data faster, and current workers configuration is fixed at 4 for stability issues and thread management bottle necks
* No mutexes or thread safe mechanism is required as these threads only reads the data from the particular index
* While the dataloader uses randomsampler to generate random sequence of indexes and generate random index batch

## Models
* Model.h holds three models as explained earlier
* Each model has three methods
	1. `Forward()` this method processes the data through network
	2. `load()` loads the weights into network
	3. `save()` saves the weights of the network
* Accuracy - as expected the tranfer learning network gives the highest accuracy of 82% while the network made from scratch produces the result upto 35-40%

## Transforms
* Uses opencv methods to preprocess the image 
* Currently these transforms are implemented
	* Resize
	* CenterCrop
	* RandomRotate
	* RandomHorizontalFlip
	* Convert_to_Tensor (converts opencv image to tensor)

## Main Program
* main program asks the user for the choice of the network
* constructs the user choice network
* asks for the operational mode i.e. train,test or evaluate
* if training or testing is chosen then it constructs dataset, dataloader and optimizer
* calls the templated train function and passes all the training parameters to it
* It calls templated evaluation method and scans the evaluation folder and predicts dog breed for each image in evaluation mode
* also the vgg-16 full pretrained network won't be trained or tested it can only evaluate

## Precompiled Headers
* I have used the precompiled headers in Visual Studio to reduce the build times from 1-2 minutes to almost instantaneous
* Pch.h will be saved as precompiled headers
* reducing build times helps to improve productivity time and testing time
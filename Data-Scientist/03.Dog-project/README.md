## Project Overview

In this project, we will build a classification model that may be applied to process real-world, user-supplied images.  Given an image of a dog, our algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.

![Dog Breed Prediction](sample_dog_output.png)

We will use different existing models for human face detection as well as for dog identification. We will compare different architectures (simple CNN architecture and Transfer Learning) in order to predict a dog breed for a given picture (if a human or a dog is detected). 

## Project Structure

```
- bottleneck-features # folder to store bottleneck features for Transfer Learning

- dog_app_files # stores images for dog_app.html

- figures # stores visualizations from dog_app.ipynb

- haarcascades # opencv-cascade-classifier

- images # default images to test classification

- custom_img # custom images to test classification

- requirements # project requirements and installation (see more details bellow)

- saved_models # CNN and Transfer Learning models

- extract_bottleneck_features.py # contains code to use pre-trained ImageNet models as feature extractors for transfer learning

- dog_app.ipynb # main notebook of the project
```

## Main Results

To evaluate all the models we use **accuracy score**, the [most famous classification performance indicator](https://arxiv.org/pdf/2008.05756.pdf). Accuracy returns an overall measure of how much the model is correctly predicting the classification of a single individual above the entire set of data.

1. To detect human faces we use **OpenCV**'s implementation of **Haar feature-based cascade classifiers**. They show a perfect accuracy in detecting human faces (100%), however, they were also able to detect humans on dog pictures (in 10% of cases).

2. To detect dogs we use a pre-trained **ResNet-50** model. It performs better than the previous human detector: 100% of dogs detected on dogs pictures; 0% dogs detected on human pictures. 

3. Finally, to detect dog breeds we use three different models:

    - Our first model uses **CNN**s with different layers and resulted in 4.67% accuracy.

    - Our second model uses Transfer Learning with bottleneck features of the **VGG-16** model; it results in 42.58% accuracy.

    - Our third model uses Transfer Learning as well, but with bottleneck features of the **Inception** model; it results in 77.75% accuracy.

## Libraries used

```
opencv-python==3.2.0.6
h5py==2.6.0
matplotlib==2.0.0
numpy==1.12.0
scipy==0.18.1
tqdm==4.11.2
keras==2.0.2
scikit-learn==0.18.1
pillow==4.0.0
ipykernel==4.6.1
tensorflow==1.0.0
```

### Run the project on the local machine, V1

Create (and activate) a new environment.

- __Linux__ (to install with __GPU support__, change `requirements/dog-linux.yml` to `requirements/dog-linux-gpu.yml`): 

```
conda env create -f requirements/dog-linux.yml
source activate dog-project
```

- __Mac__ (to install with __GPU support__, change `requirements/dog-mac.yml` to `requirements/dog-mac-gpu.yml`):

```
conda env create -f requirements/dog-mac.yml
source activate dog-project
```

**NOTE:** Some Mac users may need to install a different version of OpenCV

```
conda install --channel https://conda.anaconda.org/menpo opencv3
```

- __Windows__ (to install with __GPU support__, change `requirements/dog-windows.yml` to `requirements/dog-windows-gpu.yml`):

```
conda env create -f requirements/dog-windows.yml
activate dog-project
```

### Run the project on the local machine, V2

If V1 throws errors, try this __alternative__ to create your environment.

- __Linux__ or __Mac__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):

```
conda create --name dog-project python=3.5
source activate dog-project
pip install -r requirements/requirements.txt
```

**NOTE:** Some Mac users may need to install a different version of OpenCV

```
conda install --channel https://conda.anaconda.org/menpo opencv3
```

- __Windows__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):

```
conda create --name dog-project python=3.5
activate dog-project
pip install -r requirements/requirements.txt
```

### IPython kernel

- Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `dog-project` environment.

```
python -m ipykernel install --user --name dog-project --display-name "dog-project"
```

Open the notebook.

```
jupyter notebook dog_app.ipynb
```

- Before running code, change the kernel to match the dog-project environment by using the drop-down menu (**Kernel > Change kernel > dog-project**). Then, follow the instructions in the notebook.

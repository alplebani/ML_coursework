# COURSEWORK Alberto Plebani (ap2387)

README containing instructions on how to run the code for the coursework.

The repository can be cloned with 
```shell
git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/M2_Assessment/ap2387.git
```


# Anaconda 

The conda environment can be created using the [environment.yml](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/M2_Assessment/ap2387/-/blob/main/environment.yml?ref_type=heads) file, which contains all the packages needed to run the code:
```shell
conda env create --name CDT_ML --file environment.yml
```

# Report

The final report is presented in [ap2387.pdf](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/M2_Assessment/ap2387/-/blob/main/report/ap2387.pdf?ref_type=heads). The file is generated using LaTeX, but all LaTeX-related files are not being committed as per the instructions on the [.gitignore](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/M2_Assessment/ap2387/-/blob/main/.gitignore?ref_type=heads) file.

# Code structure

The codes to run the exercises can be found in the ```src``` folder, whereas the file [Helpers/HelperFunctions.py](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/M2_Assessment/ap2387/-/blob/main/Helpers/HelperFunctions.py?ref_type=heads) contains the definition of the classes used in the code.

The code can be run with multiple ```parser``` options, which can be accessed with the following command
```shell
python src/main.py -h
```

The options are the following:
- ```--name```: This value (str) determines the name of the model, which will be used to save the sampled images under the name 'name_ddpm_sample_0001.png' and the model 'name_ddpm_mnist-pth'
- ```--plots```: this flag determines whether you want to visualise the plots or only save them
- ```-n, --nepochs```: this determines the number of epochs used for the training. By default this value is set to 100
- ```--delta```: this value (float) determines the delta for the early stopping. By default this value is set to 0.0005
- ```--patience```: this value (int) determines how many epochs are used to evaluate the early stopping. By default this value is set to 15
- ```-t, --type```: The options here are ```DDPM``` or ```Personal```. This determines which degradation model you want to use, whether the standard one (former) or the personal one (latter). By default this value is set to 'DDPM'
- ```-l, --layers```: list of space-separated nodes in each hidden layer. By default this value is set to ```8 8 ```, meaning two hidden layers with 8 nodes each.
- ```-e, --eta```: This value (float) determines the learning rate. By default it is set to 2e-4
- ```-b,--beta```: List of two floats between 0 and 1 which determines how the noise is added and then removed. They have to be in increasing order. By default this values are ```1e-4 0.02```
- ```--nT```: This value (int) determines how many discrete steps are used to add and remove noise from the images. By default this value is set to 1000
- ```-d, --drop```: This value (float) determines the dropout rate for the pixels in the image. This is to be used only if ```type == 'Personal'```. By default this value is set to 0.2
- ```-r, --range```: List of two increasing order floats between -1 and 1 which determine the range in which the pixel luminance is adjusted

The code will then display the hyperparameters and the parser options, and it will ask the user if he's happy with those choices. If yes, type ```y```, otherwise type any other key and the code will exit.

The model will be saved in the ```model``` folder, whereas the folders ```Plots``` and ```contents``` will contain the loss function plots and the sampled images, respectively.

LELEC2885 - Image Processing and Computer Vision : Project in Image Segmentation

This project is divided in two parts. Part I treats with our implementation of UNetLight, a streamlined version of the UNet architecture.
We've trained it on the Oxford-IIIT Pet Dataset (https://www.robots.ox.ac.uk/~vgg/data/pets/). Once the dataset is uploaded in the 'Dataset' folder, it is possible to evaluate our (trained) model by running the 'main.py' file. This returns our model's predictions for the test set, which you can visualize in the folder 'Results\DefaultExp\Test'. It also computes quantitative metrics about our model's performance, which you can visualize by running the 'plot.ipynb' jupyter notebook.
It is also possible to train our network again, possibly with other parameters as the ones we've chosen. To do so, uncomment the lines concerning training in the 'main.py' file. To change the parameters (learning rate, batch size, patience...), modify the 'Todo_List\DefaultExp.yaml' file.

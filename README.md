LELEC2885 - Image Processing and Computer Vision : Project in Image Segmentation

This project is divided in two parts. Part I treats with our implementation of UNetLight, a streamlined version of the UNet architecture.
We've trained it on the Oxford-IIIT Pet Dataset (https://www.robots.ox.ac.uk/~vgg/data/pets/). It is possible to evaluate our (trained) model by running the 'main.py' file. This returns our model's predictions for the test set, which you can visualize in the folder 'Results\DefaultExp\Test'. It also computes quantitative metrics about our model's performance, which you can visualize by running the 'plot.ipynb' jupyter notebook.

Part II treats with the estimation of the model's uncertainty. Evaluating the model's uncertainty can be done by running the 'main.py' file. This returns our model's uncertainty for the test set, which you can visualize in the folder 'Results\DefaultExp\Test'.

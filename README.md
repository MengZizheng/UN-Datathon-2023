# Folder layout

- dataset:

./dataset/CMIP5/CMIP5.input.36mn.1861_2001.nc: the input of the CMIP5 dataset.

./dataset/CMIP5/CMIP5.label.nino34.12mn_3mv.1863_2003.nc: the label of the CMIP5 dataset.

./dataset/SODA/SODA.input.36mn.1871_1970.nc: the input of the SODA dataset.

./dataset/SODA/SODA.label.nino34.12mn_3mv.1873_1972.nc: the label of the SODA dataset.

./dataset/GODAS/GODAS.input.36mn.1980_2015.nc: the input of the GODAS dataset.

./dataset/GODAS/GODAS.label.12mn_3mv.1982_2017.nc: the label of the GODAS dataset.


- figure:

the figures.


- model:

the parameters of the models.


- pred_label:

the ensembled predict result and the true label.

- preds:

the predict result of every model, the number of model is 4.


- code.py:

the code. 


- README.md
README.md

# Where the data came from?
SODA, https://climatedataguide.ucar.edu/climate-data/soda-simple-ocean-data-assimilation
GODAS, https://www.esrl.noaa.gov/psd/data/gridded/data.godas.html
ERA-Interim, https://apps.ecmwf.int/datasets/data/interim-full-daily
NMME phase 1, https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/
The CMIP5 database, https://esgf-node.llnl.gov/projects/cmip5/

# What is mindspore?
mindspore, https://gitee.com/mindspore

# What difficulties did we encounter?
It is a very difficult task to predict Niso3.4, which has two main difficulties. 
First, it is difficult to extract geographical data features
Second, the total amount of data is small

# how did we solve them
We choose to use CNN model to let the neural network learn geographical features.
For the problem with a small amount of data, we use transfer learning.
First use the simulated data of CMIP5 for pre-training. Let the neural network learn the geographical features.
Then use SODA data for fine tuning to adapt the model to the real situation. 
In order to test the effect of the model and eliminate the autocorrelation of the time series, we delayed the verification set for ten years
Finally found that our model had a high correlation coefficient for Niso3.4 after 1.5 years, as shown in the figure. So we get a good Niso3.4 prediction model

# Which paper do we did we refer to?
https://www.nature.com/articles/s41586-019-1559-7

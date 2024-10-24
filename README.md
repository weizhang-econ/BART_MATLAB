# Coding Bayesian Additive Regression Trees Using MATLAB
This repository contains MATLAB code for Bayesian Additive Regression Trees (BART). While many machine learning packages in R and Python offer efficient implementations, as an econometrician, I find it useful to code these algorithms from scratch, as it deepens my understanding of the model's underlying structure.

# Files and Data
- `Boston.m`: main file for the forecasting practice
- `BART.m`: function for the posterior sampling of BART
- `fitBART.m`: function for computing fitted values for training dataset based on the sampled trees
- `fitBART_test.m`: function for computing for testing dataset based on the sampled trees
- `BLM.m`: function for the posterior sampling of a Bayesian linear regression model
- `BostonHousing.csv`: dataset used
- `Utility`: a folder contains useful functions

# Notes
I used 5 trees in the code and compared the forecasting performance of BART to OLS and Bayesian linear regression. You can change the number of trees by changing the value of `m`. I followed Chipman et al. (2010) for priors. For simplicity, I only allow three moves - growing, pruning and changing (no swapping), and they have equal probabilities (1/3). 

The root mean squared error (RMSE) using BART is around 4.73, while using OLS and linear regression produce RMSEs around 5.12.

# References
Chipman, H. A., George, E. I., & McCulloch, R. E. (2010). BART: Bayesian additive regression trees. The Annals of Applied Statistics, 4(1), 266.

# Disclaimer
This code comes without technical support of any kind. Under no circumstances will the author be held responsible for any use (or misuse) of this code in any way.


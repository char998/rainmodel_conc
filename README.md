The repository for my MSc thesis - *ES25: A Simplified Conceptual Convective Precipitation Model*


**GB84** : the code of the GB84 model (Georgakakos and Bras 1984). It contains the python file for the physical model, the Kalman filter implementation and two test notebooks: one for testing only the physical core of GB84 and one for testing and comparing the stochastic formulation of GB84 WITH different assimilation windows

**ES25** : The model developed in the spectre of this thesis. The final model is loosely based in the GB84 model in terms of the model state, but there are significant changes concerning convection and cloud evolution, water removal mechanisms and the evaporation scheme.

**data**: The folder contains data for Netherlands corresponding to July 2024, which were used to assess the model performance. The data come from different sources: ground (Cesar Network and Parsivel disdrometer) and remote sensing (multi-pointing MWR) from Cabauw, as well the reanalysis data from ERA5 corresponding to the closest grid point to Cabauw.

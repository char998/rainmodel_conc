The repository for my MSc thesis - *ES25: A Simplified Conceptual Convective Precipitation Model*
### 📁 Project Structure Overview

#### **1. GB84**
- **Origin**: Based on the *A hydrologically useful station precipitation model: 1. Formulation.*, Georgakakos and Bras (1984)
- **Contents**:
  - **GB84.py**: The Python implementation of the physical core of the model
  - **kalman_filter.py**: The kalman filter module for data assimilation.
  - Two Jupyter Notebooks:
    - **runs_and_tests.ipynb**: Tests only the physical core of the GB84 model.
    - **runs_and_tests_with_kalman.ipynb**: Tests the stochastic formulation of GB84, comparing different data assimilation windows.

#### **2. ES25**
- **Overview**: A more recent and significantly modified model developed as part of this thesis work. While loosely based on GB84 (especially regarding the state variables), it introduces notable changes in several physical processes.
- **Improvements Over GB84**:
  - Enhanced **convection and cloud evolution** schemes.
  - Revised **water removal mechanisms** (e.g., precipitation processes).
  - Updated **evaporation scheme**.
- **Notebook**:
  - **ES25_era_test.ipynb**: Demonstrates the ES25 model performance using ERA5 data for **July 2024**.

#### **3. data**
- **Purpose**: Supports model testing and evaluation.
- **Coverage**: Focused on the **Netherlands**, specifically **Cabauw**, for **July 2024**.
- **Data Sources**:
  - **Ground-based**:
    - Cesar Network
    - Parsivel Disdrometer
  - **Remote sensing**:
    - Multi-pointing Microwave Radiometer (MWR)
  - **Reanalysis**:
    - ERA5 data for the grid point closest to Cabauw

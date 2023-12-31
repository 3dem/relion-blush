# Blush Refinement

This repository contains code for running Blush regularization in RELION-5. 
This project is designed to work in conjunction with tools for cryo-electron microscopy (cryo-EM) three-dimensional reconstruction. 
For specific functionalities, we leverage the capabilities of RELION, an open-source, highly flexible and high-performance solution for cryo-EM.

## Related Resources

For detailed information about RELION and its features, please see the [RELION repository here](https://github.com/3dem/relion).

## Installation and Usage

Before proceeding with the installation of this project, ensure you have RELION installed and configured correctly. 
If you have not yet done so, please refer to the [RELION GitHub repository](https://github.com/3dem/relion) for installation instructions.

## Model Weights

After installation of this project by running `pip insall .` in the root directory, the model weights will automatically be 
downloaded and installed from this [Zenodo record](https://zenodo.org/records/10072731) during the first run of the command-line util. 

## Issues and Bug Reports

Please refer to the [RELION repository issue tracking](https://github.com/3dem/relion/issues) for reporting issues or bugs related to Blush.

## Acknowledgements

The work in this repository is based on the research conducted in the paper 
"Data-driven regularisation lowers the size barrier of cryo-EM structure determination" by Kimanius, Dari, et al. 
The methodologies and algorithms presented in this paper are implemented in the code found within this repository. 

For the complete details of the research and methods that inspired this work, please refer to the following publication:

Kimanius, D., et al. (2023) Data-driven regularisation lowers the size barrier of cryo-EM structure determination. bioRxiv. 
DOI: [10.1101/2023.10.23.563586](https://doi.org/10.1101/2023.10.23.563586).


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

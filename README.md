## Reproduce this work

Clone the project from the Github repository ([esbenkc/alexnet](https://github.com/esbenkc/alexnet)). You can do this by running `git clone https://github.com/esbenkc/alexnet`. When it is downloaded, run `hipcc main.cpp -std=c++11 -o alexnet.out` and `sbatch alexnet.sh` if running on lux.ucsc.edu, otherwise run the compiled alexnet.out program however you wish. If running on lux, it will generate output files called nn.err and nn.out. Hopefully, nn.err contains nothing while nn.out contains the relevant outputs from the project, as showcased in this report.

To reproduce the Tensorflow implementation of deep CNNs prior to this GPU implementation of a linear neural network, visit [esbenkc/tinyimage-classification](https://github.com/esbenkc/tinyimage-classification).
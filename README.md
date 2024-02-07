# Code for the "Multi-Aspect ECG Generation Using Hierarchical Variational Autoencoders" Paper

cNVAE-ECG is a model based on [NVAE](https://github.com/NVlabs/NVAE) for conditional generation of 12 main-lead ECG signals longer than one heartbeat or _multi-aspect_ ECGs.

<p align="center">
    <img src="img/cNVAE-ECG.png" width="550">
</p>

#### To train cNVAE-ECG model:
0. Download [PTB-XL dataset](https://physionet.org/content/ptb-xl/1.0.3/);
1. Install dependencies from the ```requirements.txt``` file;
2. Go to the *scripts* directory and run the ```training_conditional.sh``` script with specified parameters;
3. After training, run evaluation with the ```evaluating_conditional.sh``` script with specified parameters.

#### To run testing on downstream task:
1. Go to the *train_test* directory and clone code from the [ecg_ptbxl_benchmarking repository](https://github.com/helme/ecg_ptbxl_benchmarking);
2. Also clone code with our conditional implementation of the [WaveGAN* and Pulse2Pulse models](https://anonymous.4open.science/r/Pulse2Pulse-5E0F/README.md) into *train_test* directory;
3. For comparing enrichment of the entire training dataset quality, run the ```run_testing.sh``` script with specified parameters for each method;
4. For comparing enrichment of the pathology class quality, run the ```run_testing_ones.sh``` script with specified parameters for each method.

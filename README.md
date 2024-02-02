# Code for the "Multi-Aspect ECG Generation Using Hierarchical Variational Autoencoders" Paper

#### To train cNVAE-ECG model:
0. Download [PTB-XL dataset](https://physionet.org/content/ptb-xl/1.0.3/);
1. Install dependencies from the ```requirements.txt``` file;
2. Go to the *scripts* directory and run the ```training_conditional.sh``` script with specified parameters;
3. After training, run evaluation with the ```evaluating_conditional.sh``` script with specified parameters.

#### To run testing on downstream task:
1. Go to the *train_test* directory and clone code from the [ecg_ptbxl_benchmarking repository](https://github.com/helme/ecg_ptbxl_benchmarking);
2. Also clone code with conditional implementation of the [WaveGAN* and Pulse2Pulse models](https://anonymous.4open.science/r/Pulse2Pulse-5E0F/README.md) into *train_test* directory;
3. For comparing enrichment of the entire training dataset quality, run the ```run_testing.sh``` script with specified parameters for each method;
4. For comparing enrichment of the pathology class quality, run the ```run_testing_ones.sh``` script with specified parameters for each method.

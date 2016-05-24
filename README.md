# semantic-design-languages
Code for our IDETC 2016 paper - "How Designs Differ: Non-Linear Embeddings Illuminate Intrinsic Design Complexity"

To prepare the parametric space for hyperparameter optimization: python parametric_space.py

To train a model using optimized hyperparameters and evaluate with testing samples: python training.py

Edit parameters in config.ini

The settings of the kernel PCA and autoencoders are in the configuration files in './hp-opt'

The configuration files are named 'hp_<example>_<sample size>_<semantic space dimensionality>_<indices of testing samples>.ini'

We used [Spearmint](https://github.com/HIPS/Spearmint) for hyperparameter optimization of kernel PCA and autoencoders
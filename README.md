# design_embeddings_idetc_2016
Experiment code associated with our IDETC 2016 paper: "How Designs Differ: Non-Linear Embeddings Illuminate Intrinsic Design Complexity"

To prepare the parametric space for hyperparameter optimization: python parametric_space.py

To train a model using optimized hyperparameters and evaluate with testing samples: python training.py

Edit parameters in config.ini

The settings of the kernel PCA and autoencoders are in the configuration files in './hp-opt'

The configuration files are named 'hp\_\<example\>\_\<sample size\>\_\<semantic space dimensionality\>\_\<indices of testing samples\>.ini'

We used [Spearmint](https://github.com/HIPS/Spearmint) for hyperparameter optimization of kernel PCA and autoencoders

The code is licensed under the MIT license. Feel free to use all or portions for your research or related projects so long as you provide the following citation information:

Wei Chen, Jonah Chazan, and Mark Fuge.  "How Designs Differ: Non-linear Embeddings Illuminate Intrinsic Design Complexity," for Proceedings of ASME 2016 International Design Engineering Technical Conferences & Computers and Information in Engineering Conference, August 21-24, 2016, Charlotte, USA.

    @inproceedings{chen:how_designs_differ_idetc2016,
        author = {Chen, Wei and Chazan, Jonah and Fuge, Mark},
        title = {How Designs Differ: Non-linear Embeddings Illuminate Intrinsic Design Complexity},
        booktitle = {ASME International Design Engineering Technical Conferences},
        year = {2016},
        month = {August},
        location = {Charlotte, USA},
        publisher = {ASME}
    }

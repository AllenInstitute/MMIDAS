# MMIDAS (Mixture Model Inference with Discrete-coupled AutoencoderS)

A generalized and unsupervised mixture variational model with a multi-armed deep neural network, to jointly infer the discrete type and continuous type-specific variability. This framework can be applied to analysis of both, uni-modal and multi-modal datasets. It outperforms comparable models in inferring interpretable discrete and continuous representations of cellular identity, and uncovers novel biological insights. MMIDAS can thus help researchers identify more robust cell types, study cell type-dependent continuous variability, interpret such latent factors in the feature domain, and study multi-modal datasets.


# Data
- [Allen Institute Mouse Smart-seq dataset](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE115746)
- [Allen Institute Mouse 10x isocortex dataset](https://assets.nemoarchive.org/dat-jb2f34y)
- [Allen Institute Patch-seq data](https://dandiarchive.org/dandiset/000020/)

  The electrophysiological features have been computaed according to the approach provided in [cplAE_MET/preproc/data_proc_E.py](cplAE_MET/preproc/data_proc_E.py)
- [Seattle Alzheimer’s disease dataset (SEA-AD)](https://SEA-AD.org/)

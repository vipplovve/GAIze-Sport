# Data Sources Used

The models were trained & fine-tuned using real-world match footage. This document outlines the sources used for training.

## Primary Video Sources
The detection and action recognition models were trained on frames and sequences extracted from the following Creative Commons (CC BY) sources:

* **Football:** [MLS Next Cup | Barcelona Residency vs. Toronto FC | U-15 Highlights](https://www.youtube.com/watch?v=RAhgpBvMF38).
* **Basketball:** [Nehru World School vs St. Andrews Agra | U17 CBSE Basketball Semi-Final](https://www.youtube.com/watch?v=K_IWQHiCgzE).

## Distribution Policy
To keep this repository lightweight and to prioritize the privacy of the athletes involved, we **do not distribute** the raw training frames, annotated datasets, or coordinate tensors.

* **Included:** This repository contains the final **trained model weights (.pt/.pth files)**. These weights represent the patterns learned during training and are ready for inference.
* **Not Included:** The raw image data and training logs are omitted. Users interested in replicating the training environment can do so by extracting frames from the original sources listed above using our provided scripts.

## Credits
We gratefully acknowledge the creators—**lozzzproductionz** and **SportsAlgo**—for making their high-quality match coverage available via Creative Commons.



# Starting-kit for Efficient Burst HDR and Restoration @NTIRE2025

\[[Homepage](https://cvlai.net/ntire/2025//)\] \[[Codalab](https://codalab.lisn.upsaclay.fr/competitions/21201)\]

- [Starting-kit for Efficient Burst HDR and Restoration @NTIRE2025](#starting-kit-for-efficient-burst-hdr-and-restoratio-ntire2015)
  - [Overview](#overview)
  - [Tips](#tips)



## Overview

The Efficient Burst HDR and Restoration is geared towards training neural networks for fusing multi-frame raw images into high-quality sRGB image in scenarios where multi-frame raw have different brightness levels.

In this starting kit, we will provide you with a possible solution, but you don't have to follow this approach.

Additionally, we will also provide you with tips on important considerations during the competition and the submission process.

In the [`code_example/tutorial.ipynb`](code_example/tutorial.ipynb), we provide examples and notes on reading data, lite ISP, calculating scores, and submission.

In the [`evaluate`](evaluate), we provide the validation code that we submitted on Codalab.

## Tips

- You are **NOT** restricted to train their algorithms only on the provided dataset. Other **PUBLIC** dataset
  can be used as well. However, you need to mention in the final submitted factsheet what public datasets you have used.
- Please ensure that your testing process can be conducted on a single NVIDIA RTX 3090 (i.e., the memory usage needs to be less than 24GB). This is to limit resource usage during deployment.
-  We will check the participants' code after the final test stage to ensure fairness.



# Dataset attribution and terms

WSN-DS by Iman Almomani, Bassam Al-Kasasbeh and Mousa AL-Akhras.
Paper: https://doi.org/10.1155/2016/4731953
Authors’ public dataset listing: https://www.kaggle.com/datasets/bassamkasasbeh1/wsnds

The dataset describes NS-2 simulated wireless sensor network traffic with Normal,
Blackhole, Grayhole, Flooding and Scheduling classes. It is public research data;
it is not captured production telecom traffic.

The Kaggle listing states “Data files © Original Authors”. The paper is open access
under Creative Commons Attribution, but that does not establish an MIT license
for the CSV. Consult the dataset authors’ terms before redistribution or commercial
use. This repository distributes no WSN-DS rows. MIT covers this repository’s new
code only. The synthetic demo is generated locally by scikit-learn and is not WSN-DS.

Download: `python scripts/download_dataset.py`. If the endpoint requires sign-in,
use the Kaggle website’s secure sign-in and save the CSV to `data/WSN-DS.csv`.
Never commit credentials, raw data, medical data, or large model files.

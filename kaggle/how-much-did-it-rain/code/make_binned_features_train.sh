#!/bin/sh

python binned_features.py '../processed/train_' 'Reflectivity' 17 200 20 &
python binned_features.py '../processed/train_' 'Composite' 17 200 20 &
python binned_features.py '../processed/train_' 'DistanceToRadar' 17 200 20 &
python binned_features.py '../processed/train_' 'HybridScan' 17 200 20 &
python binned_features.py '../processed/train_' 'RR1' 17 200 20 &
python binned_features.py '../processed/train_' 'RR2' 17 200 20 &
python binned_features.py '../processed/train_' 'RR3' 17 200 20 &

wait

python binned_features.py '../processed/train_' 'TimeToEnd' 17 200 20 &
python binned_features.py '../processed/train_' 'RadarQualityIndex' 17 200 20 &
python binned_features.py '../processed/train_' 'ReflectivityQC' 17 200 20 &
python binned_features.py '../processed/train_' 'RhoHV' 17 200 20 &
python binned_features.py '../processed/train_' 'Velocity' 17 200 20 &
python binned_features.py '../processed/train_' 'Zdr' 17 200 20 & 
python binned_features.py '../processed/train_' 'LogWaterVolume' 17 200 20 &

wait

python binned_features.py '../processed/train_' 'Reflectivity' 7 18 10 &
python binned_features.py '../processed/train_' 'Composite' 7 18 10 &
python binned_features.py '../processed/train_' 'DistanceToRadar' 7 18 10 &
python binned_features.py '../processed/train_' 'HybridScan' 7 18 10 &
python binned_features.py '../processed/train_' 'RR1' 7 18 10 &
python binned_features.py '../processed/train_' 'RR2' 7 18 10 &
python binned_features.py '../processed/train_' 'RR3' 7 18 10 & 

wait

python binned_features.py '../processed/train_' 'TimeToEnd' 7 18 10 &
python binned_features.py '../processed/train_' 'RadarQualityIndex' 7 18 10 &
python binned_features.py '../processed/train_' 'ReflectivityQC' 7 18 10 &
python binned_features.py '../processed/train_' 'RhoHV' 7 18 10 &
python binned_features.py '../processed/train_' 'Velocity' 7 18 10 &
python binned_features.py '../processed/train_' 'Zdr' 7 18 10 & 
python binned_features.py '../processed/train_' 'LogWaterVolume' 7 18 10 &

wait


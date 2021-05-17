#!/bin/sh

#this is pretty simple way to this so 
python binned_features.py '../processed/test_' 'Reflectivity' 17 200 20 &
python binned_features.py '../processed/test_' 'Composite' 17 200 20 &
python binned_features.py '../processed/test_' 'DistanceToRadar' 17 200 20 &
python binned_features.py '../processed/test_' 'HybridScan' 17 200 20 &
python binned_features.py '../processed/test_' 'RR1' 17 200 20 &
python binned_features.py '../processed/test_' 'RR2' 17 200 20 &
python binned_features.py '../processed/test_' 'RR3' 17 200 20 &

wait

python binned_features.py '../processed/test_' 'TimeToEnd' 17 200 20 &
python binned_features.py '../processed/test_' 'RadarQualityIndex' 17 200 20 &
python binned_features.py '../processed/test_' 'ReflectivityQC' 17 200 20 &
python binned_features.py '../processed/test_' 'RhoHV' 17 200 20 &
python binned_features.py '../processed/test_' 'Velocity' 17 200 20 &
python binned_features.py '../processed/test_' 'Zdr' 17 200 20 &
python binned_features.py '../processed/test_' 'LogWaterVolume' 17 200 20 &

wait

python binned_features.py '../processed/test_' 'Reflectivity' 7 18 10 &
python binned_features.py '../processed/test_' 'Composite' 7 18 10 &
python binned_features.py '../processed/test_' 'DistanceToRadar' 7 18 10 &
python binned_features.py '../processed/test_' 'HybridScan' 7 18 10 &
python binned_features.py '../processed/test_' 'RR1' 7 18 10 &
python binned_features.py '../processed/test_' 'RR2' 7 18 10 &
python binned_features.py '../processed/test_' 'RR3' 7 18 10 &

wait
python binned_features.py '../processed/test_' 'TimeToEnd' 7 18 10 &
python binned_features.py '../processed/test_' 'RadarQualityIndex' 7 18 10 &
python binned_features.py '../processed/test_' 'ReflectivityQC' 7 18 10 &
python binned_features.py '../processed/test_' 'RhoHV' 7 18 10 &
python binned_features.py '../processed/test_' 'Velocity' 7 18 10 &
python binned_features.py '../processed/test_' 'Zdr' 7 18 10 &
python binned_features.py '../processed/test_' 'LogWaterVolume' 7 18 10 &
wait



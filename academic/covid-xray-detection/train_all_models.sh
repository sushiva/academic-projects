#!/bin/bash

cd /home/bhargav/academic-projects/academic/covid-xray-detection

echo "Training Model 1: ANN with RGB Images"
python src/train.py --config config/model1_ann_rgb.yaml

echo ""
echo "Training Model 2: ANN with Grayscale Images"
python src/train.py --config config/model2_ann_grayscale.yaml

echo ""
echo "Training Model 3: ANN with Gaussian-blurred Images"
python src/train.py --config config/model3_ann_blur.yaml

echo ""
echo "Training Model 4: ANN with Laplacian-Filtered Images"
python src/train.py --config config/model4_ann_laplacian.yaml

echo ""
echo "All models trained successfully!"

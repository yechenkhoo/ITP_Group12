## Flow:

### Training, validating and testing all models
1. Make necessary changes to ```Classes/ModelFactory.py```
2. Go to ```poseModelSimplified.py```
    1. set the ```path_csv``` to the generated csv with poses coordinates.
    2. change ```test_run_name``` to new name (folder name e.g. C1, C2)
    3. set the number of ```epochs```
```
python poseModelSimplified.py
```

### Results analysis
3. Go to ```utils/convert_csv_table.py```.
    1. change folder name in the file to the new name (folder name from 2.2)
    2. run to get the results table in image format:
```
python utils/convert_csv_table.py
```

4. Go to ```utils/generate_analysis.py```.
    1. identify which model you want to generate analyis for.
    2. change ```folder```, ```model``` and ```value``` to the correct names.
    3. run to get a visualisation of what positions are frequently confused by the model.
```
python utils/generate_analysis.py
```

5. Go to ```utils/PlotResults.py```.
    1. identify which model you want to generate visualisations for.
    2. change ```dataset_dir```, ```csv_path``` and ```output_dir``` to correct values.
    3. run to get a visualisation of image, pose detected and model predictions.
```
python utils/PlotResults.py
```
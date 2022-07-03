

### CUDA generation model training: 
```
sh train_generate_model.sh
```
CUDA models save in ```./cuda_model```

### Train Filter Models:
#### Classifier Filter
```
sh train_classifier_filter.sh
```

#### Slot-Gate Classifier Filter
```
sh train_slot-gate_classifier_filter.sh
```

### CUDA Generation
```
sh run_generation.sh
```
CUDA data saves in ```./cuda_data```

### Run Evaluation
Before evaulation, make sure that coco-dst data is generated via 
```
cd ../coco-dst
sh run_eval_data.sh
```

Then Train Trippy, Trade or simpleTOD model. For more detail, check out README.md in those directories.


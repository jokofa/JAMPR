# JAMPR

This is the repository accompanying the paper 
[Learning to Solve Vehicle Routing Problems with Time Windows through Joint Attention](https://arxiv.org/abs/2006.09100)

The paper is currently under review and we only provide our data generator code and the test and validation sets. 
Model code will be published after the review period. 

Please cite us: 
```
@article{falkner2020learning,
  title={Learning to Solve Vehicle Routing Problems with Time Windows through Joint Attention},
  author={Falkner, Jonas K and Schmidt-Thieme, Lars},
  journal={arXiv preprint arXiv:2006.09100},
  year={2020}
}
``` 
### Dependencies

* Python>=3.6
* NumPy


### Generating Data
```bash
python data_generator.py --problem cvrptw --name test --seed 1234 --service_window 1000 --service_duration 10
```


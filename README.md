# DPCBM
Code for private Concept Learning of Concept Bottleneck Models and Concept Embedding Models with Differential Privacy

### To train
```
python3 train.py +data='cub' +model='cbm'

where +data is {'cub', 'awa2', 'mnist'} and +model is {'cbm','cem'}
```

### To Test
```
python3 test.py +data='cub' +model='cbm'
where +data is {'cub', 'awa2', 'mnist'} and +model is {'cbm','cem'}
```

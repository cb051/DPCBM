

## Experiments

#### To Train CBM and CEM (non private models)
uses three arguments: `+data`, `+model`, and `+mode`, and `+ckpt` 

where 
1. `+data` is {'mnist', 'awa2', 'cub200'}
2. `+model` is {'cem', 'cbm'}
3. `+ckpt` is a path to a saved checkpoint

e.g.,

1. Run `sbatch train.sh +data='mnist' +model='cbm'` to train a model
2. Run `sbatch test.sh +data='mnist' +model='cbm' +ckpt='path/to/dir'` to see metrics for a trained model

---

#### To Train or Test DP CBM and CEM (private models)
uses arguments `+data`, `+model`, `+epsilon`, and `+ckpt` 

where 
1. `+data` is {'mnist', 'awa2', 'cub200'} 
2. `+model` is {'dp_cem', 'dp_cbm'}, 
3. `+epsilon`is {0.01,0.1,1,10,100} or an arbitrary value in [0.01,100]
3. `+ckpt` is a path to a saved checkpoint

e.g.,
1. Run `sbatch train.sh +data='mnist' +model='dp_cbm' +epsilon=0.1` to train a model
2. Run `sbatch test.sh +data='mnist' +model='dp_cbm' +epsilon=0.1 +ckpt` to see metrics for a trained model
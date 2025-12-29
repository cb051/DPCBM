# DPCBM
Code for private Concept Learning of Concept Bottleneck Models and Concept Embedding Models with Differential Privacy

### Motivations
In sensitive domains, where both privacy-preservation and model interpretability are desirable, differentially private noise can degrade the quality of explanations. Explanations given as concepts in Concept Learning Models (CLMs) are susceptible to spurious relationships which can be further exacerbated under privacy constraints. This work proposes orthogonal weight regularization as a method for mitigating disparate impact by differentially private noise on Concept Learning Models. 

The effects of noise has only been narrowly considered for concept-based models, mainly in the context of noisy or corrupted inputs. To the best of our knowledge, this is the first work considering the effect of DP on CLMs. We provide fine-grained analysis of the effect of gaussian noise on Concept Learning Models and show that a collapse in concept quality can occur even under very small amounts of noise and lead to concept leakage. We also show that improving orthogonal regularization in convolutional layers is a strong direction for improving the utility of private models and mitigating disparate impact by DP SGD.

### Usage

#### To train
```
python3 train.py +data='cub' +model='cbm'
```
where +data is {'cub', 'awa2', 'mnist'} and +model is {'cbm','cem', 'dp_cbm', 'dp_cem'}

DP settings can be adjusted in `configs/model/dp_cbm.yaml` or `dp_cem.yaml`.

#### To Test
```
python3 test.py +data='cub' +model='cbm'
```
where +data is {'cub', 'awa2', 'mnist'} and +model is {'cbm','cem','dp_cbm', 'dp_cem'}

DP settings can be adjusted in `configs/model/dp_cbm.yaml` or `dp_cem.yaml`

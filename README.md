# DPCBM
Code for private Concept Learning of Concept Bottleneck Models and Concept Embedding Models with Differential Privacy

### Motivations
In sensitive domains, where privacy-preservation and model interpretability are desirable, we intuit that differentially private noise will degrade the quality of explanations. As the concepts in CLMs are based on latent features, they are susceptible to "concept leakage" where a concept behaves purely as a hidden layer bottleneck. This reference to leakage is different from privacy leakage in that concept leakage occurs when the concept set prioritizes end task prediction over linearly separable concepts. We refer exclusively to concept leakage throughout this work as leakage.

The effects of noise has only been narrowly considered for concept-based models, mainly in the context of noisy or corrupted inputs. The impact of differentially private noise has, however, been considered for feature-based and inherently interpretable systems. We consider a separate case post-hoc explanations are provided using latent features as opposed to end task predictions. 

We provide fine-grained analysis of the effect of gaussian noise on Concept Learning Models; Namely, we show that a collapse in concept quality can occur even under very small amounts of noise and leads to concept leakage and show that improving orthogonal regularization in convolutional layers is a strong direction for improving the utility of private models and mitigating disparate impact by DP SGD.

### Usage

#### To train
```
python3 train.py +data='cub' +model='cbm'
```
where +data is {'cub', 'awa2', 'mnist'} and +model is {'cbm','cem'}

#### To Test
```
python3 test.py +data='cub' +model='cbm'
```
where +data is {'cub', 'awa2', 'mnist'} and +model is {'cbm','cem'}

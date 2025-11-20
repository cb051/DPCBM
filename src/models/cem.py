import numpy as np
import lightning as L
import torch

import torch.nn.functional as f

import torchvision.models as models
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import sklearn.metrics

from torchmetrics.wrappers import BootStrapper
from torchmetrics.classification import Accuracy, MultilabelAccuracy
from scipy.stats import norm

import hydra

import datasets.mnist_dataset as mnist_ds
import datasets.awa2_dataset as awa2_ds
import datasets.cub200_dataset as cub200_ds

# adapted from Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off
# Zarlenga et al. 2022 from Koh et al. 2020 Concept Bottleneck Models
# ==================================================

# A version of ConceptBottleneckModel but with altered _run_step for Opacus library
class DP_CBM(L.LightningModule):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        concept_loss_weight=0.01,
        task_loss_weight=1,

        extra_dims=0,
        bool=False,
        sigmoidal_prob=True,
        sigmoidal_extra_capacity=True,
        bottleneck_nonlinear=None,
        output_latent=False,

        x2c_model=None,
        c_extractor_arch=None,
        c2y_model=None,
        c2y_layers=None,

        optimizer="adam",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        weight_loss=None,
        task_class_weights=None,
        top_k_accuracy=None,
    ):
        """
        Constructs a joint Concept Bottleneck Model (CBM) as defined by
        Koh et al. 2020.

        :param int n_concepts: The number of concepts given at training time.
        :param int n_tasks: The number of output classes of the CEM.
        :param int emb_size: The size of each concept embedding. Defaults to 16.
        :param str embedding_activation: A valid nonlinearity name to use for the
            generated embeddings. It must be one of [None, "sigmoid", "relu",
            "leakyrelu"] and defaults to "leakyrelu".
        :param Bool shared_prob_gen: Whether or not weights are shared across
            all probability generators. Defaults to True.
        :param float concept_loss_weight: Weight to be used for the final loss'
            component corresponding to the concept classification loss. Default
            is 0.01.
        :param float task_loss_weight: Weight to be used for the final loss'
            component corresponding to the output task classification loss.
            Default is 1.

        :param Pytorch.Module c2y_model:  A valid pytorch Module used to map the
            CEM's bottleneck (with size n_concepts * emb_size) to `n_tasks`
            output activations (i.e., the output of the CEM).
            If not given, then a simple leaky-ReLU MLP, whose hidden
            layers have sizes `c2y_layers`, will be used.
        :param List[int] c2y_layers: List of integers defining the size of the
            hidden layers to be used in the MLP to predict classes from the
            bottleneck if c2y_model was NOT provided. If not given, then we will
            use a simple linear layer to map the bottleneck to the output classes.
        :param Fun[(int), Pytorch.Module] c_extractor_arch: A generator function
            for the latent code generator model that takes as an input the size
            of the latent code before the concept embedding generators act (
            using an argument called `output_dim`) and returns a valid Pytorch
            Module that maps this CEM's inputs to the latent space of the
            requested size.

        :param str optimizer:  The name of the optimizer to use. Must be one of
            `adam` or `sgd`. Default is `adam`.
        :param float momentum: Momentum used for optimization. Default is 0.9.
        :param float learning_rate:  Learning rate used for optimization.
            Default is 0.01.
        :param float weight_decay: The weight decay factor used during
            optimization. Default is 4e-05.
        :param List[float] weight_loss: Either None or a list with n_concepts
            elements indicating the weights assigned to each predicted concept
            during the loss computation. Could be used to improve
            performance/fairness in imbalanced datasets.
        :param List[float] task_class_weights: Either None or a list with
            n_tasks elements indicating the weights assigned to each output
            class during the loss computation. Could be used to improve
            performance/fairness in imbalanced datasets.

        :param List[int] top_k_accuracy: List of top k values to report accuracy
            for during training/testing when the number of tasks is high.
        """
        super().__init__()
        self.n_concepts = n_concepts
        self.output_latent = output_latent

        if x2c_model is not None:
            # Then this is assumed to be a module already provided as
            # the input to concepts method
            self.x2c_model = x2c_model
        else:
            self.x2c_model = c_extractor_arch(
                output_dim=(n_concepts + extra_dims)
            )

        # Now construct the label prediction model
        if c2y_model is not None:
            # Then this method has been provided to us already
            self.c2y_model = c2y_model
        else:
            # Else we construct it here directly
            units = [n_concepts + extra_dims] + (c2y_layers or []) + [n_tasks]
            layers = []
            for i in range(1, len(units)):
                layers.append(torch.nn.Linear(units[i-1], units[i]))
                if i != len(units) - 1:
                    # layers.append(torch.nn.BatchNorm1d(128))
                    # layers.append(torch.nn.Dropout(0.5))
                    layers.append(torch.nn.LeakyReLU())
            self.c2y_model = torch.nn.Sequential(*layers)

        # For legacy purposes, we wrap the model around a torch.nn.Sequential
        # module
        self.sig = torch.nn.Sigmoid()
        if sigmoidal_extra_capacity:
            # Keeping this for backwards compatability
            bottleneck_nonlinear = "sigmoid"
        if bottleneck_nonlinear == "sigmoid":
            self.bottleneck_nonlin = torch.nn.Sigmoid()
        elif bottleneck_nonlinear == "leakyrelu":
            self.bottleneck_nonlin = torch.nn.LeakyReLU()
        elif bottleneck_nonlinear == "relu":
            self.bottleneck_nonlin = torch.nn.ReLU()
        elif (bottleneck_nonlinear is None) or (
            bottleneck_nonlinear == "identity"
        ):
            self.bottleneck_nonlin = lambda x: x
        else:
            raise ValueError(
                f"Unsupported nonlinearity '{bottleneck_nonlinear}'"
            )

        self.loss_concept = torch.nn.BCELoss(weight=weight_loss)
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights)
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(
                pos_weight=task_class_weights
            )
        )
        self.bool = bool
        self.concept_loss_weight = concept_loss_weight
        self.task_loss_weight = task_loss_weight
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.extra_dims = extra_dims
        self.top_k_accuracy = top_k_accuracy
        self.n_tasks = n_tasks
        self.sigmoidal_prob = sigmoidal_prob
        self.sigmoidal_extra_capacity = sigmoidal_extra_capacity
        
    def _unpack_batch(self, batch):
        x = batch[0]
        if isinstance(batch[1], list):
            y, c = batch[1]
        else:
            y, c = batch[1], batch[2]
        if len(batch) > 3:
            competencies = batch[3]
        else:
            competencies = None

        return x, y, (c, competencies)
    
    def _forward(
        self,
        x,
        competencies=None,
        c=None,
        y=None,
        train=False,
        latent=None,
        output_latent=None,
        output_embeddings=False,
    ):
        output_latent = (
            output_latent if output_latent is not None
            else self.output_latent
        )
        if latent is None:
            latent = self.x2c_model(x)
        if self.sigmoidal_prob or self.bool:
            if self.extra_dims:
                # Then we only sigmoid on the probability bits but
                # let the other entries up for grabs
                c_pred_probs = self.sig(latent[:, :-self.extra_dims])
                c_others = self.bottleneck_nonlin(latent[:,-self.extra_dims:])
                c_pred =  torch.cat([c_pred_probs, c_others], axis=-1)
                c_sem = c_pred_probs
            else:
                c_pred = self.sig(latent)
                c_sem = c_pred
        else:
            # Otherwise, the concept vector itself is not sigmoided
            # but the semantics
            c_pred = latent
            if self.extra_dims:
                c_sem = self.sig(latent[:, :-self.extra_dims])
            else:
                c_sem = self.sig(latent)
        
            pos_embeddings = torch.unsqueeze(pos_embeddings, dim=-1)
            neg_embeddings = torch.unsqueeze(neg_embeddings, dim=-1)

        if self.bool:
            y = self.c2y_model((c_pred > 0.5).float())
        else:
            y = self.c2y_model(c_pred)
        tail_results = []

        if output_latent:
            tail_results.append(latent)
        if output_embeddings:
            tail_results.append(pos_embeddings)
            tail_results.append(neg_embeddings)
        return tuple([c_sem, c_pred, y] + tail_results)
    
    def forward(
        self,
        x,
        c=None,
        y=None,
        latent=None,
        competencies=None,
    ):
        return self._forward(
            x,
            train=False,
            c=c,
            y=y,
            competencies=competencies,
            latent=latent,
        )
    
    def _run_step(self,
                  batch,
                  batch_idx,
                  train=False,
                  ):
        x, y, (c, competencies) = self._unpack_batch(batch)
        outputs = self._forward(
            x,
            c=c,
            y=y,
            train=train,
            competencies=competencies,
        )
        c_sem, c_logits, y_logits = outputs[0], outputs[1], outputs[2]
        if self.task_loss_weight != 0:
            task_loss = self.loss_task(
                y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1),
                y,
            )
            task_loss_scalar = task_loss.detach()
        else:
            task_loss = 0
            task_loss_scalar = 0
        if self.concept_loss_weight != 0:
            # We separate this so that we are allowed to
            # use arbitrary activations (i.e., not necessarily in [0, 1])
            # whenever no concept supervision is provided
            # Will only compute the concept loss for concepts whose certainty
            # values are fully given

            concept_loss = self.loss_concept(c_sem, c)
            concept_loss_scalar = concept_loss.item()
            loss = self.concept_loss_weight * concept_loss.item() + task_loss + \
                self._extra_losses(
                    x=x,
                    y=y,
                    c=c,
                    c_sem=c_sem,
                    c_pred=c_logits,
                    y_pred=y_logits,
                    competencies=competencies,
                )

        else:
            loss = task_loss + self._extra_losses(
                x=x,
                y=y,
                c=c,
                c_sem=c_sem,
                c_pred=c_logits,
                y_pred=y_logits,
                competencies=competencies
            )
            concept_loss_scalar = 0.0
        # compute accuracy
        (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1) = compute_accuracy(
            c_sem,
            y_logits,
            c,
            y,
        )
        result = {
            "c_accuracy": c_accuracy,
            "c_auc": c_auc,
            "c_f1": c_f1,
            "y_accuracy": y_accuracy,
            "y_auc": y_auc,
            "y_f1": y_f1,
            "concept_loss": concept_loss_scalar,
            "task_loss": task_loss_scalar,
            "loss": loss.detach(),
            "avg_c_y_acc": (c_accuracy + y_accuracy) / 2,
        }
        if self.top_k_accuracy is not None:
            y_true = y.reshape(-1).cpu().detach()
            y_pred = y_logits.cpu().detach()
            labels = list(range(self.n_tasks))
            if isinstance(self.top_k_accuracy, int):
                top_k_accuracy = list(range(1, self.top_k_accuracy))
            else:
                top_k_accuracy = self.top_k_accuracy

            for top_k_val in top_k_accuracy:
                if top_k_val:
                    y_top_k_accuracy = sklearn.metrics.top_k_accuracy_score(
                        y_true,
                        y_pred,
                        k=top_k_val,
                        labels=labels,
                    )
                    result[f'y_top_{top_k_val}_accuracy'] = y_top_k_accuracy
        return loss, result
    
    def training_step(self, batch, batch_no):
        loss, result = self._run_step(batch, batch_no, train=True)
        for name, val in result.items():
            if self.n_tasks <= 2:
                prog_bar = (
                    ("auc" in name) or
                    ("mask_accuracy" in name) or
                    ("current_steps" in name) or
                    ("num_rollouts" in name)
                )
            else:
                prog_bar = (
                    ("c_auc" in name) or
                    ("y_accuracy" in name) or
                    ("mask_accuracy" in name) or
                    ("current_steps" in name) or
                    ("num_rollouts" in name)

                )
            self.log(name, val, prog_bar=prog_bar)
        return {
            "loss": loss,
            "log": {
                "c_accuracy": result['c_accuracy'],
                "c_auc": result['c_auc'],
                "c_f1": result['c_f1'],
                "y_accuracy": result['y_accuracy'],
                "y_auc": result['y_auc'],
                "y_f1": result['y_f1'],
                "concept_loss": result['concept_loss'],
                "task_loss": result['task_loss'],
                "loss": result['loss'],
                "avg_c_y_acc": result['avg_c_y_acc'],
            },
        }

    def validation_step(self, batch, batch_no):
        _, result = self._run_step(batch, batch_no, train=False)
        for name, val in result.items():
            if self.n_tasks <= 2:
                prog_bar = (("auc" in name))
            else:
                prog_bar = (("c_auc" in name) or ("y_accuracy" in name))
            self.log("val_" + name, val, prog_bar=prog_bar)
        result = {
            "val_" + key: val
            for key, val in result.items()
        }
        return result

    def test_step(self, batch, batch_no):
        loss, result = self._run_step(batch, batch_no, train=False)
        for name, val in result.items():
            self.log("test_" + name, val, prog_bar=True)
        return result['loss']

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "loss",
        }

# adapted from Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off
# Zarlenga et al. 2022
# ==================================================
#  A version of CEM that implements Opacus DP make_private
class DPConceptEmbeddingModel(DP_CBM):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        emb_size=16,
        embedding_activation="leakyrelu",
        shared_prob_gen=True,
        concept_loss_weight=1,
        task_loss_weight=1,

        c2y_model=None,
        c2y_layers=None,
        c_extractor_arch=None,
        output_latent=False,

        optimizer="adam",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        weight_loss=None,
        task_class_weights=None,
        top_k_accuracy=None,

        # DP stuff
        train_dl=None,
    ):
        """
        Constructs a Concept Embedding Model (CEM) as defined by
        Espinosa Zarlenga et al. 2022.
        :param int n_concepts: The number of concepts given at training time.
        :param int n_tasks: The number of output classes of the CEM.
        :param int emb_size: The size of each concept embedding. Defaults to 16.
        :param str embedding_activation: A valid nonlinearity name to use for the
            generated embeddings. It must be one of [None, "sigmoid", "relu",
            "leakyrelu"] and defaults to "leakyrelu".
        :param Bool shared_prob_gen: Whether or not weights are shared across
            all probability generators. Defaults to True.
        :param float concept_loss_weight: Weight to be used for the final loss'
            component corresponding to the concept classification loss. Default
            is 0.01.
        :param float task_loss_weight: Weight to be used for the final loss'
            component corresponding to the output task classification loss.
            Default is 1.

        :param Pytorch.Module c2y_model:  A valid pytorch Module used to map the
            CEM's bottleneck (with size n_concepts * emb_size) to `n_tasks`
            output activations (i.e., the output of the CEM).
            If not given, then a simple leaky-ReLU MLP, whose hidden
            layers have sizes `c2y_layers`, will be used.
        :param List[int] c2y_layers: List of integers defining the size of the
            hidden layers to be used in the MLP to predict classes from the
            bottleneck if c2y_model was NOT provided. If not given, then we will
            use a simple linear layer to map the bottleneck to the output classes.
        :param Fun[(int), Pytorch.Module] c_extractor_arch: A generator function
            for the latent code generator model that takes as an input the size
            of the latent code before the concept embedding generators act (
            using an argument called `output_dim`) and returns a valid Pytorch
            Module that maps this CEM's inputs to the latent space of the
            requested size.

        :param str optimizer:  The name of the optimizer to use. Must be one of
            `adam` or `sgd`. Default is `adam`.
        :param float momentum: Momentum used for optimization. Default is 0.9.
        :param float learning_rate:  Learning rate used for optimization.
            Default is 0.01.
        :param float weight_decay: The weight decay factor used during
            optimization. Default is 4e-05.
        :param List[float] weight_loss: Either None or a list with n_concepts
            elements indicating the weights assigned to each predicted concept
            during the loss computation. Could be used to improve
            performance/fairness in imbalanced datasets.
        :param List[float] task_class_weights: Either None or a list with
            n_tasks elements indicating the weights assigned to each output
            class during the loss computation. Could be used to improve
            performance/fairness in imbalanced datasets.

        :param List[int] top_k_accuracy: List of top k values to report accuracy
            for during training/testing when the number of tasks is high.
        """
        L.LightningModule.__init__(self)
        self.n_concepts = n_concepts

        self.pre_concept_model = c_extractor_arch(output_dim=None)
       
        self.output_latent = output_latent

        self.task_loss_weight = task_loss_weight
        self.concept_context_generators = torch.nn.ModuleList()
        self.concept_prob_generators = torch.nn.ModuleList()
        self.shared_prob_gen = shared_prob_gen
        self.top_k_accuracy = top_k_accuracy
        for i in range(n_concepts):
            if embedding_activation == "leakyrelu":
                self.concept_context_generators.append(
                    torch.nn.Sequential(*[
                        torch.nn.Linear(
                            list(
                                self.pre_concept_model.modules()
                            )[-1].out_features,
                            # Two as each concept will have a positive and a
                            # negative embedding portion which are later mixed
                            2 * emb_size,
                        ),
                        torch.nn.LeakyReLU(),
                    ])
                )
            else:
                raise ValueError("leakyrelu is only implemented here. change accordingly")
            if self.shared_prob_gen and (
                len(self.concept_prob_generators) == 0
            ):
                # Then we will use one and only one probability generator which
                # will be shared among all concepts. This will force concept
                # embedding vectors to be pushed into the same latent space
                self.concept_prob_generators.append(torch.nn.Linear(
                    2 * emb_size,
                    1,
                ))
            elif not self.shared_prob_gen:
                self.concept_prob_generators.append(torch.nn.Linear(
                    2 * emb_size,
                    1,
                ))
        if c2y_model is None:
            # Else we construct it here directly
            units = [
                n_concepts * emb_size
            ] + (c2y_layers or []) + [n_tasks]
            layers = []
            for i in range(1, len(units)):
                layers.append(torch.nn.Linear(units[i-1], units[i]))
                if i != len(units) - 1:
                    layers.append(torch.nn.LeakyReLU())
            self.c2y_model = torch.nn.Sequential(*layers)
        else:
            self.c2y_model = c2y_model
        self.sig = torch.nn.Sigmoid()
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights)
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(
                weight=task_class_weights
            )
        )
        self.concept_loss_weight = concept_loss_weight
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.n_tasks = n_tasks
        self.emb_size = emb_size


        # vars for DP setup
        self.train_dl = train_dl
        self.loss_before_dp = torch.nn.BCELoss(weight=weight_loss)
        self.loss_concept = None

        # note accuracies/f1/auc are calculated in _run_step(). Accuracy is 
        # recalculated here for ease
        self.y_accuracy = Accuracy(task="multiclass", num_classes=self.n_tasks)
        self.c_accuracy = MultilabelAccuracy(num_labels=self.n_concepts)
        self.concept_bootstrapper = BootStrapper(self.c_accuracy,
                                                 num_bootstraps=100,
                                                 sampling_strategy="multinomial", #sample over entire dataset
                                                 quantile=torch.tensor([0.025, 0.975]) # 95% CI
                                                 )
        self.task_bootstrapper = BootStrapper(self.y_accuracy,
                                                 num_bootstraps=100,
                                                 sampling_strategy="multinomial", #sample over entire dataset
                                                 quantile=torch.tensor([0.025, 0.975]) # 95% CI
                                                 )
        
        self.automatic_optimization = False # TURN OFF Auto Optimization FOR DP
        self.ckpt_version = None
    
    def orthogonal_regularization(self, y_pred, y):
        # look at orthogonality in convolutional layers
        # return penalty term
        fro_norms = []
        for name, params in self.named_parameters():
            if ".conv" in name:
                # calculate dot product AA^T
                mat = torch.matmul(params, torch.swapaxes(params,-1,-2)).cpu() - torch.eye(params.shape[0],params.shape[1]).cpu()
                # mat = torch.matmul(params, params.transpose(2,3)).cpu() - torch.eye(params.shape[-1], params.shape[-2]).cpu() # (AA^T) - I
                fro_norms = torch.linalg.norm(mat**2)
                
        loss = torch.mean(torch.cat(fro_norms,0))

        return 0.1 * loss

    def _forward(
        self,
        x,
        c=None,
        y=None,
        train=False,
        latent=None,
        competencies=None,
        output_embeddings=False,
        output_latent=None,
        ):
        output_latent = (
            output_latent if output_latent is not None
            else self.output_latent
        )
        if latent is None:
            pre_c = self.pre_concept_model(x)
            contexts = []
            c_sem = []

            # First predict all the concept probabilities
            for i, context_gen in enumerate(self.concept_context_generators):
                if self.shared_prob_gen:
                    prob_gen = self.concept_prob_generators[0]
                else:
                    prob_gen = self.concept_prob_generators[i]
                context = context_gen(pre_c)
                prob = prob_gen(context)
                contexts.append(torch.unsqueeze(context, dim=1))
                c_sem.append(self.sig(prob))
            c_sem = torch.cat(c_sem, axis=-1)
            contexts = torch.cat(contexts, axis=1)
            latent = contexts, c_sem
        else:
            contexts, c_sem = latent

        # Then, time to do the mixing between the positive and the
        # negative embeddings
        # Then time to mix!
        c_pred = (
            contexts[:, :, :self.emb_size] * torch.unsqueeze(prob, dim=-1) +
            contexts[:, :, self.emb_size:] * (1 - torch.unsqueeze(prob, dim=-1))
        )
        c_pred = c_pred.view((-1, self.emb_size * self.n_concepts))
        y = self.c2y_model(c_pred)
        tail_results = []
        
        if output_latent:
            tail_results.append(latent)
        if output_embeddings:
            tail_results.append(contexts[:, :, :self.emb_size])
            tail_results.append(contexts[:, :, self.emb_size:])
        return tuple([c_sem, c_pred, y] + tail_results)

    # DP step, setup opacus components (optimizer, privacy_engine, loss
    def _setup_dp(self):
        # https://discuss.pytorch.org/t/pytorch-lightning-support/113507/7
        print("Setting up Differential Privacy...")
        print(f"""DP parameters: epsilon={config.model.privacy_args.epsilon}, delta={config.model.privacy_args.delta}, max_grad_norm={config.model.privacy_args.max_grad_norm}""")
        privacy_engine = PrivacyEngine()
        # use diff optimizer for head of pre_concept_model)
        optimizer = torch.optim.SGD([{'params': self.pre_concept_model.parameters()}, # head only for training
                                     ], lr=config.model.model_args.lr, momentum=0.9) # used to form self.concept_optimizer
        try:
            _, optimizer_gc, loss_gc, self.train_dl = privacy_engine.make_private_with_epsilon(
                module=self.pre_concept_model,
                optimizer=optimizer, 
                data_loader=self.train_dl,
                target_epsilon=config.model.privacy_args.epsilon,
                target_delta=config.model.privacy_args.delta,
                max_grad_norm=config.model.privacy_args.max_grad_norm,
                epochs=config.data.epochs,
                poisson_sampling=False,
                grad_sample_mode="ghost", # reduces memory overhead in grad calc step
                criterion=self.loss_before_dp
            )
            print("Successfully initialized PrivacyEngine")
        except Exception as e:
            print(f"Failed to initialize PrivacyEngine: {e}")
            raise e
        
        # update arguments passed to fit
        return optimizer_gc, loss_gc, privacy_engine
    
    def setup(self, stage):
        # https://discuss.pytorch.org/t/pytorch-lightning-support/113507/7
        # setup DP before training
        if config.model.train_private:
            self.x2c_optimizer, self.loss_concept, self.privacy_engine = self._setup_dp()
        else:
            self.x2c_optimizer = torch.optim.SGD(self.x2c.model[0][1].parameters(),
                            lr=config.model.model_args.lr,
                            momentum=0.9)
            self.loss_concept = self.loss_before_dp
    
    # DP step, log epsilon
    def _get_epsilon(self):
        try:
            budget = self.privacy_engine.get_epsilon(1e-5)
            return budget
        except:
            return -1

    def configure_optimizers(self):
        self.general_optimizer = torch.optim.SGD([
                                                {'params': self.concept_context_generators.parameters()},
                                                {'params': self.concept_prob_generators.parameters()},
                                                {'params': self.c2y_model.parameters()}], lr=config.model.model_args.lr, momentum=0.9)
        x2c_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                self.concept_optimizer,
                                verbose=True,
                            )
        general_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                self.general_optimizer,
                                verbose=True,
                            )
        return [self.concept_optimizer,self.general_optimizer], [x2c_lr_scheduler,general_lr_scheduler]

    def on_train_epoch_end(self):
        sch1, sch2 = self.lr_schedulers()
        sch1.step(self.trainer.callback_metrics["loss"])
        sch2.step(self.trainer.callback_metrics["loss"])
    
        # log epsilon at end of each epoch
        if config.model.train_private:
            budget = self._get_epsilon()
            self.log('privacy_budget', budget)
    
    # manual loss / optimization step
    def _calculate_loss_optimization(self, concept_loss, task_loss, orth_loss):
        combined_loss = (self.concept_loss_weight * concept_loss.item()) + (self.task_loss_weight * task_loss) + orth_loss
        self.general_optimizer.zero_grad()
        self.concept_optimizer.zero_grad()

        self.manual_backward(combined_loss)
        
        self.general_optimizer.step()
        self.concept_optimizer.step()

        return combined_loss
# calculate c,y accuracies (AUC/ROC, F1, Acc)
    def compute_accuracy(
        y_pred,
        y_true,
        binary_output=False,
    ):
        if (len(y_pred.shape) < 2) or (y_pred.shape[-1] == 1) or binary_output:
            return compute_bin_accuracy(
                y_pred=y_pred,
                y_true=y_true,
            )
        y_probs = torch.nn.Softmax(dim=-1)(y_pred).cpu().detach()
        used_classes = np.unique(y_true.reshape(-1).cpu().detach())
        y_probs = y_probs[:, sorted(list(used_classes))]
        y_pred = y_pred.argmax(dim=-1).cpu().detach()
        y_true = y_true.reshape(-1).cpu().detach()
        y_accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
        try:
            y_auc = sklearn.metrics.roc_auc_score(
                y_true,
                y_probs,
                multi_class='ovo',
            )
        except:
            y_auc = 0.0
        y_f1 = 0.0
        return (y_accuracy, y_auc, y_f1)

    # override training step in ConceptBottleneckModel
    def training_step(self,batch,batch_idx):
        x,y,c = batch
        y = y.long()
        c_sem, _, y_pred = self._forward(x=x,
                                          c=c)
        concept_loss = self.loss_concept(c_sem, c)
        task_loss = f.cross_entropy(y_pred, y)
        orth_loss = self.orthogonal_regularization(y_pred,y)

        # log c,y accuracy, f1, auc c,y loss etc.
        (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1) = _compute_accuracy(
            c_sem,
            y_pred,
            c,
            y,
        )
        self.log("concept_loss", concept_loss.item())
        self.log("task_loss", task_loss)
        self.log("loss", (concept_loss.item() + task_loss))
        self.log('orthog_loss', orth_loss,prog_bar=True)
        self.log("c_accuracy", c_accuracy)
        self.log("y_accuracy", y_accuracy)
        self.log("c_auc", c_auc)
        self.log("y_auc", y_auc)
        self.log("c_f1", c_f1)
        self.log("y_f1", y_f1)


        results = self._calculate_loss_optimization(concept_loss, task_loss, orth_loss)
        return results
    
    def on_validation_end(self):
        # manual optimization seems to break logging
        version_path = f"{self.trainer.default_root_dir}/lightning_logs/version_{self.logger.version}"
        checkpoint_path = f"{version_path}/checkpoints/epoch-{self.current_epoch}.ckpt"

        # save every 10 epochs
        if (self.current_epoch % 10 == 0) or (self.current_epoch == self.trainer.max_epochs-1):
            self.trainer.save_checkpoint(checkpoint_path)
        
        # keep config info in a readme
        with open(f'{version_path}/readme.md', 'w') as f:
            f.write(f"model: {config.model_name}")
            if config.model.train_private:
                f.write(f"epsilon: {config.model.privacy_args.epsilon}")

    def test_step(self, batch, batch_idx):
        x,y,c = batch # always in form x,y,c
        y = y.long()

        # predict
        c_pred, _, y_pred = self._forward(x=x,
                                          c=c)
        
        # loss + accuracy
        x2c_loss = self.loss_concept(c_pred, c)
        c2y_loss = f.cross_entropy(y_pred, y)
        orth_loss = self.orthogonal_regularization(y_pred,y)

        combined_loss = (self.concept_loss_weight * x2c_loss.item()) + (self.task_loss_weight * c2y_loss) + orth_loss

        # confidence intervals
        self.concept_bootstrapper.update(c_pred, c)
        self.task_bootstrapper.update(y_pred, y)

        #log stuff
        self.log('x2c loss', x2c_loss.item(), prog_bar=True)
        self.log('c2y_loss', c2y_loss,prog_bar=True)
        self.log('combined_loss', combined_loss, prog_bar=True)
        
        return combined_loss

    def on_test_epoch_end(self):
        self.concept_bootstrapper.quantile = self.concept_bootstrapper.quantile.to(self.device)
        self.task_bootstrapper.quantile = self.task_bootstrapper.quantile.to(self.device)
        c_bs = self.concept_bootstrapper.compute()
        y_bs = self.task_bootstrapper.compute()

        # calculate CI + accuracies
        self.log('C acc', (c_bs['mean']*100), prog_bar=True)
        self.log('Y acc', (y_bs['mean']*100), prog_bar=True)
        print(f"=== 95% CI concept: {c_bs['quantile'][0] * 100} {c_bs['quantile'][1] * 100} ===")
        print(f"=== 95% CI task: {y_bs['quantile'][0] * 100} {y_bs['quantile'][1] * 100} ===")

#=======================================
#  main ================================
#=======================================
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main():
    # set dataloaders
    if config.data.dataset_name == 'mnist':
        train_dl, val_dl = mnist_ds.MNIST().get_dataloaders(batch_size=config.batch_size, NDW=config.num_workers)
    elif config.data.dataset_name == 'cub200':
        train_dl, val_dl = cub200_ds.CUB().get_dataloaders(batch_size=config.batch_size, NDW=config.num_workers)
    elif config.data.dataset_name == 'awa2':
        train_dl, val_dl = awa2_ds.get_dataloaders(batch_size=config.batch_size, NDW=config.num_workers)

    dp_cem = DPConceptEmbeddingModel(n_concepts=config.data.num_concepts, # Number of training-time concepts. MNIST has 7, CUB has 112
                                    n_tasks=config.data.num_classes, # Number of output labels. MNIST has 10, CUB has 200
                                    emb_size=config.model.model_args.latent_emb_size,  # We will use an embedding size of 128
                                    concept_loss_weight=config.model.model_args.concept_loss_weight,  # The weight assigned to the concept prediction loss relative to the task predictive loss.
                                    task_loss_weight=config.model.model_args.task_loss_weight,
                                    learning_rate=config.model.model_args.lr,  # The learning rate to use during training.
                                    c_extractor_arch=config.model.model_args.c_extractor_arch, # Here we provide our generating function for the latent code generator model.
                                    c2y_model=None,  # We will let the API simply add a linear layer from the concept bottleneck to the downstream task labels
                                    shared_prob_gen=False, # note: shared gradients are incompatible w opacus library
                                    train_dl=train_dl,
                                    )

    # setup model (name local logger paths according to experiment)
    if config.model.train_private: # train DP version wo solution
        logger_path = f"{config.model.logging.local_path}/{config.model.logging.ckpt_path_name}_" \
            f"{config.data.dataset_name}_e{config.model.privacy_args.epsilon}"
    elif config.model.train_private and config.model.train_private.use_orthog_loss: # train DP version w orthog loss
         logger_path = f"{config.model.logging.local_path}/{config.model.logging.ckpt_path_name}_" \
            f"{config.data.dataset_name}_e{config.model.privacy_args.epsilon}_orthog_loss"
    elif config.model.train_private and config.model.train_private.use_orthog_clipping: # train DP version w orthog weight clipping
        logger_path = f"{config.model.logging.local_path}/{config.model.logging.ckpt_path_name}_" \
            f"{config.data.dataset_name}_e{config.model.privacy_args.epsilon}_orthog_clipping"
    else: # train non DP version
        logger_path = f"{config.model.logging.local_path}/{config.model.logging.ckpt_path_name}_{config.data.dataset_name}"

    if config.model.logging.log_local:
        # log metrics locally
        lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')
        logger = L.pytorch.loggers.CSVLogger(logger_path)
        save_every_10_epochs = L.pytorch.callbacks.ModelCheckpoint(
                                                                every_n_epochs=10, 
                                                                save_top_k=-1,
                                                                save_last=True
                                                                ) # manual logging required when using automatic optimization
    else: 
        lr_monitor = None
        save_every_10_epochs = None

    trainer = L.Trainer(
                accelerator=config.accelerator,  # Change to "cpu" if you are not running on a GPU!
                devices=config.devices, 
                max_epochs=1 if config.one_epoch_run else config.data.epochs,  # The number of epochs we will train our model for
                check_val_every_n_epoch=1,  # And how often we will check for validation metrics
                default_root_dir=logger_path if config.model.logging.log_local else None,
                callbacks=[lr_monitor,save_every_10_epochs],
                logger=logger,
                )

    # train model
    if config.train:
        trainer.fit(dp_cem, train_dl, val_dl)
    elif config.test:
        trainer.test(config.ckpt,val_dl)
    
if __name__ == '__main__':
    main()
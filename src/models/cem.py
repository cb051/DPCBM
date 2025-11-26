import numpy as np
import lightning as L
import torch

import torch.nn.functional as f

import torchvision.models as models
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import sklearn.metrics

from torchmetrics.wrappers import BootStrapper
from torchmetrics.classification import Accuracy, MultilabelAccuracy, AUROC, F1Score

# local packages
import utils.resnet34 as res34
import warnings  # ignore user warnings
import utils.orthog_reg_loss as ot_loss

# adapted from Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off
# Zarlenga et al. 2022
# ==================================================
#  A version of CEM that implements Opacus DP make_private
class ConceptEmbeddingModel(L.LightningModule):
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
        config=None,

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

        self.y_accuracy = Accuracy(task="multiclass", num_classes=self.n_tasks)
        self.c_accuracy = MultilabelAccuracy(num_labels=self.n_concepts)
        # concept / task auc + f1
        self.concept_auc = AUROC(task="multilabel", num_labels=config.data.num_concepts)
        self.task_auc = AUROC(task="multiclass", num_classes=config.data.num_classes)
        self.concept_f1 = F1Score(task="multilabel", num_labels=config.data.num_concepts)
        self.task_f1 = F1Score(task="multiclass", num_classes=config.data.num_classes)
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
        self.concept_auc_bs = BootStrapper(self.concept_auc,
                                                 num_bootstraps=100,
                                                 sampling_strategy="multinomial", #sample over entire dataset
                                                 quantile=torch.tensor([0.025, 0.975], device=self.device) # 95% CI
                                                 )
        self.task_auc_bs = BootStrapper(self.task_auc,
                                                 num_bootstraps=100,
                                                 sampling_strategy="multinomial", #sample over entire dataset
                                                 quantile=torch.tensor([0.025, 0.975], device=self.device) # 95% CI
                                                 )
        
        self.automatic_optimization = False # TURN OFF Auto Optimization FOR DP
        self.ckpt_version = None
        self.config = config
    
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

    def log_metrics(self,accs):
        c_pred = c = y_pred = y = None
        if accs is not None:
            c_pred, c, y_pred, y = accs

        # Handle log_only: Compute and log metrics without updating
        if accs is not None:
            # Log metrics
            c_accuracy = self.c_accuracy(c_pred, c)
            y_accuracy = self.y_accuracy(y_pred, y)
            concept_auc = self.concept_auc(c_pred, c.long())
            task_auc = self.task_auc(y_pred, y)
            concept_f1 = self.concept_f1(c_pred, c)
            task_f1 = self.task_f1(y_pred, y)

            self.log('task_accuracy (y)', y_accuracy)
            self.log('concept_accuracy (c)', c_accuracy)
            self.log('concept_auc', concept_auc)
            self.log('task_auc', task_auc)
            self.log('concept_f1', concept_f1)
            self.log('task_f1', task_f1)
    
    
    # DP step, setup opacus components (optimizer, privacy_engine, loss
    def _setup_dp(self):
        # https://discuss.pytorch.org/t/pytorch-lightning-support/113507/7
        print("Setting up Differential Privacy...")
        print(f"""DP parameters: epsilon={self.config.model.privacy_args.epsilon}, delta={self.config.model.privacy_args.delta}, max_grad_norm={self.config.model.privacy_args.max_grad_norm}""")
        if not ModuleValidator.is_valid(self.pre_concept_model):
            print("Model not compatible with Opacus, attempting to fix...")
            self.pre_concept_model = ModuleValidator.fix(self.pre_concept_model)
            print("Model fixed for Opacus compatibility")
        privacy_engine = PrivacyEngine()
        # use diff optimizer for head of pre_concept_model)
        optimizer = torch.optim.SGD([{'params': self.pre_concept_model.parameters()}, # head only for training
                                     ], lr=self.config.model.model_args.lr, momentum=0.9) # used to form self.concept_optimizer
        try:
            _, optimizer_gc, _ = privacy_engine.make_private_with_epsilon(
                module=self.pre_concept_model,
                optimizer=optimizer, 
                data_loader=self.train_dl,
                target_epsilon=self.config.model.privacy_args.epsilon,
                target_delta=self.config.model.privacy_args.delta,
                max_grad_norm=self.config.model.privacy_args.max_grad_norm,
                epochs=self.config.data.epochs,
                poisson_sampling=False,
                # grad_sample_mode="ghost", # reduces memory overhead in grad calc step, otherwise uses hooks
                criterion=self.loss_before_dp
            )
            loss_gc = self.loss_before_dp
            print("Successfully initialized PrivacyEngine")
        except Exception as e:
            print(f"Failed to initialize PrivacyEngine: {e}")
            raise e
        
        # update arguments passed to fit
        return optimizer_gc, loss_gc, privacy_engine
    
    def setup(self, stage):
        # https://discuss.pytorch.org/t/pytorch-lightning-support/113507/7
        # setup DP before training
        if self.config.model.train_private:
            self.concept_optimizer, self.loss_concept, self.privacy_engine = self._setup_dp()
        else:
            self.concept_optimizer = torch.optim.SGD(self.pre_concept_model.parameters(),
                            lr=self.config.model.model_args.lr,
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
                                                {'params': self.c2y_model.parameters()}], lr=self.config.model.model_args.lr, momentum=0.9)
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
        sch1.step(self.trainer.callback_metrics["combined_loss"])
        sch2.step(self.trainer.callback_metrics["combined_loss"])
    
        # log epsilon at end of each epoch
        if self.config.model.train_private:
            budget = self._get_epsilon()
            self.log('privacy_budget', budget)
    
    # manual loss / optimization step
    def _calculate_loss_optimization(self, concept_loss, task_loss, orth_loss,eval=False):
        combined_loss = None
        if self.config.model.train_private and self.config.model.privacy_args.use_orthog_loss:     
            combined_loss = (self.concept_loss_weight * concept_loss.mean()) + (self.task_loss_weight * task_loss) + orth_loss
            self.log('orthog_loss', orth_loss)
            
        else:
            combined_loss = (self.concept_loss_weight * concept_loss.mean()) + (self.task_loss_weight * task_loss)
        self.log('concept_loss', concept_loss.mean())
        self.log("task_loss", task_loss)
        self.log("combined_loss", combined_loss)
        if not eval:
            self.general_optimizer.zero_grad()
            self.concept_optimizer.zero_grad()

            self.manual_backward(combined_loss)
            
            self.general_optimizer.step()
            self.concept_optimizer.step()

        return combined_loss
    
    # override training step in DP_CBM
    def training_step(self,batch,batch_idx):
        x,y,c = batch
        y = y.long()
        c_pred, _, y_pred = self._forward(x=x,
                                          c=c)

        # log c,y accuracy, f1, auc c,y loss etc.
        self.log_metrics((c_pred,c,y_pred,y))
        concept_loss = self.loss_concept(c_pred, c)
        task_loss = f.cross_entropy(y_pred, y)
        orth_loss = None
        if self.config.model.train_private and self.config.model.privacy_args.use_orthog_loss:
            orth_loss = ot_loss.OrthogonalLoss()(gamma=self.config.model.privacy_args.orthog_gamma,
                                             model=self.pre_concept_model)
        else:
            orth_loss = -1
        combined_loss = self._calculate_loss_optimization(concept_loss=concept_loss, 
                                                          task_loss=task_loss,
                                                          orth_loss=orth_loss, 
                                                          eval=False)
        return combined_loss
    
    def validation_step(self,batch,batch_idx):
        x,y,c = batch
        y = y.long()
        c_pred, c_sem, y_pred = self._forward(x=x,
                                          c=c)

        concept_loss = self.loss_concept(c_pred, c)
        task_loss = f.cross_entropy(y_pred, y)
        orth_loss = None
        if self.config.model.train_private and self.config.model.privacy_args.use_orthog_loss:
            orth_loss = ot_loss.OrthogonalLoss()(gamma=self.config.model.privacy_args.orthog_gamma,
                                             model=self.pre_concept_model)
        else:
            orth_loss = -1
        combined_loss = self._calculate_loss_optimization(concept_loss=concept_loss, 
                                                          task_loss=task_loss,
                                                          orth_loss=orth_loss, 
                                                          eval=True)
        # log metrics
        self.log_metrics((c_pred,c,y_pred,y))
        return combined_loss

    def on_validation_end(self):
        # manual optimization seems to break logging
        version_path = f"{self.trainer.default_root_dir}/lightning_logs/version_{self.logger.version}"
        checkpoint_path = f"{version_path}/checkpoints/epoch-{self.current_epoch}.ckpt"

        # save every 10 epochs
        if (self.current_epoch % 10 == 0) or (self.current_epoch == self.trainer.max_epochs-1):
            self.trainer.save_checkpoint(checkpoint_path)
        
        # keep config info in a readme
        with open(f'{version_path}/readme.md', 'w') as f:
            f.write(f"config: {self.config}")
            # if self.config.model.train_private:
            #     f.write(f"epsilon: {self.config.model.privacy_args.epsilon}\n")
            #     f.write(f"delta: {self.config.model.privacy_args.delta}\n")
            #     f.write(f"max grad norm: {self.config.model.privacy_args.max_grad_norm}\n")
            #     if self.config.model.privacy_args.use_orthog_loss:
            #             f.write(f"use orthogonal loss: {self.config.model.privacy_args.use_orthog_loss}\n")
            #     if self.config.model.privacy_args.use_orthog_clipping:
            #             f.write(f"use orthogonal clipping: {self.config.model.privacy_args.use_orthog_clipping}\n")

    def test_step(self, batch, batch_idx):
        x,y,c = batch # always in form x,y,c
        y = y.long()

        # predict
        c_pred, c_sem, y_pred = self._forward(x=x,
                                          c=c)
        
        # loss + accuracy
        x2c_loss = self.loss_concept(c_pred, c)
        c2y_loss = f.cross_entropy(y_pred, y)
        orth_loss = None
        if self.config.model.train_private and self.config.model.privacy_args.use_orthog_loss:
            orth_loss = ot_loss.OrthogonalLoss()(gamma=self.config.model.privacy_args.orthog_gamma,
                                             model=self.pre_concept_model)
        else:
            orth_loss = -1

        combined_loss = self._calculate_loss_optimization(x2c_loss, 
                                                          c2y_loss, 
                                                          orth_loss,
                                                          eval=True)

        # confidence intervals
        self.concept_bootstrapper.update(c_pred, c)
        self.task_bootstrapper.update(y_pred, y)
        self.task_auc_bs.update(y_pred, y)
        self.concept_auc_bs.update(c_pred, c.long())
        
        return combined_loss

    def on_test_epoch_end(self):
        self.concept_bootstrapper.quantile = self.concept_bootstrapper.quantile.to(self.device)
        self.task_bootstrapper.quantile = self.task_bootstrapper.quantile.to(self.device)
        self.concept_auc_bs.quantile = self.concept_auc_bs.quantile.to(self.device)
        self.task_auc_bs.quantile = self.task_auc_bs.quantile.to(self.device)

        c_bs = self.concept_bootstrapper.compute()
        y_bs = self.task_bootstrapper.compute()
        c_auc_bs = self.concept_auc_bs.compute()
        y_auc_bs = self.task_auc_bs.compute()

        # calculate CI + accuracies
        self.log('C acc', (c_bs['mean'] * 100), prog_bar=True)
        self.log('Y acc', (y_bs['mean'] * 100), prog_bar=True)
        self.log('C AUC', (c_auc_bs['mean'] * 100), prog_bar=True)
        self.log('Y AUC', (y_auc_bs['mean'] * 100), prog_bar=True)
        
        print(f"=== 95% CI concept acc: {torch.abs((c_bs['quantile'][0] * 100)-(c_bs['mean'] * 100))} {torch.abs((c_bs['mean'] * 100)-(c_bs['quantile'][1] * 100))} ===")
        print(f"=== 95% CI task acc: {torch.abs((y_bs['quantile'][0] * 100)-(y_bs['mean'] * 100))} {torch.abs((y_bs['mean'] * 100)-(y_bs['quantile'][1] * 100))} ===")
        
        print(f"=== 95% CI concept AUC: mean: {torch.abs((c_auc_bs['mean'] * 100))} upper: {torch.abs((c_auc_bs['quantile'][0] * 100)-(c_auc_bs['mean'] * 100))} lower: {torch.abs((c_auc_bs['mean'] * 100) - (c_auc_bs['quantile'][1] * 100))} ===")
        print(f"=== 95% CI task AUC: mean: {torch.abs((y_auc_bs['mean']  * 100))} upper: {torch.abs((y_auc_bs['quantile'][0] * 100)-(y_auc_bs['mean']  * 100))} lower: {torch.abs((y_auc_bs['mean']  * 100)-(y_auc_bs['quantile'][1] * 100))} ===")

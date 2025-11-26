import torch
import matplotlib.pyplot as plt
from torch import nn
import torch
import torch.nn.functional as f


import lightning as L
import torchmetrics
from torchmetrics.wrappers import BootStrapper
import numpy as np

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import sys

import utils.resnet34 as res34
from utils.x2c import X2C
from utils.c2y import C2Y
import utils.orthog_reg_loss as ot_loss

from torchmetrics.wrappers import BootStrapper
from torchmetrics.classification import Accuracy, MultilabelAccuracy
import os


# check out for opacus help: https://openmined.org/blog/differentially-private-deep-learning-using-opacus-in-20-lines-of-code/
# how Opacus (DP libary) works: https://opacus.ai/tutorials/intro_to_advanced_features
class CBM(L.LightningModule):
    def __init__(self,
                 train_dl,
                 config):
        '''
        Concept Bottleneck Model (CBM) partially defined according
        to Koh et al. 2020 
        Constructs private and non privatized versions of a CBM. 
        This class produces the following:
        1. a non-private CBM according to Koh et al.
        2. a private CBM using DP SGD
        3. a private CBM using DP SGD and orthogonal regularization

        
        args:
        train_dl (torch.utils.data.DataLoader): used to setup opacus DP SGD optimizer
        config (dict): sets the following arguments ->
            - num_concepts (int): number of concepts to predict target task
            - num_classes (int): number of classes in target task
            - epochs (int): number of epochs
            - local_path (str): ckpt path to save logs and checkpoints
            - concept_loss_weight (float): for weighting concept loss in composite loss
            - task_loss_weight (float): for weighting task loss in composite loss
        '''
        super().__init__()

        # set x2c (concept encoder)
        if config.model.model_args.X2C == 'resnet34':
            concept_encoder = res34.resnet34_generator_model(output_dim=config.data.num_concepts)

        self.x2c = X2C(
                       num_concepts=config.data.num_concepts,
                       x2c_model=concept_encoder)
        
        self.c2y = C2Y(num_classes=config.data.num_classes,
                       num_concepts=config.data.num_concepts,
                       c2y_model=None)

        self.train_dl = train_dl
        self.loss_before_dp = torch.nn.BCEWithLogitsLoss()

        self.y_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=config.data.num_classes)
        self.c_accuracy = torchmetrics.classification.MultilabelAccuracy(num_labels=config.data.num_concepts)
        self.task_loss_weight = config.model.model_args.task_loss_weight
        self.concept_loss_weight = config.model.model_args.concept_loss_weight

        # concept / task auc + f1
        self.concept_auc = torchmetrics.AUROC(task="multilabel", num_labels=config.data.num_concepts)
        self.task_auc = torchmetrics.AUROC(task="multiclass", num_classes=config.data.num_classes)
        self.concept_f1 = torchmetrics.F1Score(task="multilabel", num_labels=config.data.num_concepts)
        self.task_f1 = torchmetrics.F1Score(task="multiclass", num_classes=config.data.num_classes)

        # for accuracy CI
        self.concept_bootstrapper = BootStrapper(self.c_accuracy,
                                                 num_bootstraps=100,
                                                 sampling_strategy="multinomial", #sample over entire dataset
                                                 quantile=torch.tensor([0.05, 0.95], device=self.device) # 95% CI
                                                 ) 
        self.task_bootstrapper = BootStrapper(self.y_accuracy,
                                                 num_bootstraps=100,
                                                 sampling_strategy="multinomial", #sample over entire dataset
                                                 quantile=torch.tensor([0.025, 0.975], device=self.device) # 95% CI
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
                                                 
        self.automatic_optimization = False # turn off automatic optimzation
        self.config = config

    def forward(self,x):
        c_pred = self.x2c(x)
        y_pred = self.c2y(c_pred)
        return c_pred, y_pred

    def constrain_orthogonal_weight(self):
        for name, params in self.x2c.named_parameters():
            if ".conv" in name and params.requires_grad:
                # reshape matrix
                sq = params.view(params.shape[0],-1).to(self.device)
                
                # bound values W/||W||_2
                W = torch.div(sq,(torch.linalg.norm(sq)**2)).to(self.device)

                # ||WW^T - I||_F
                I = torch.eye(W.shape[0], requires_grad=True).to(self.device)
                res = torch.linalg.norm(torch.matmul(W,W.T) - I).to(self.device)
                res = sq - res
                # print(f"SHAPE {params.shape}")
                # print(f"SHAPE {res.shape}")
                params.data.copy_(res.view(params.shape))

    def on_after_backward(self):
        # orthogonalize gradients
        if self.config.model.train_private and self.config.model.privacy_args.use_orthog_clipping:
            self.constrain_orthogonal_weight()


    def training_step(self, batch, batch_idx):
        #  we are given X,Y,C
        x,y,c = batch
        y = y.long()

        # pred
        c_pred, y_pred = self(x)
        # print(torch.argmax(y_pred).cpu())
        # print(y)

        # loss + accuracy + epsilon
        concept_loss = self.loss_concept(c_pred, c)
        task_loss = f.cross_entropy(y_pred, y)

        orth_loss = None
        if self.config.model.train_private and self.config.model.privacy_args.use_orthog_loss:
            orth_loss = ot_loss.OrthogonalLoss()(gamma=self.config.model.privacy_args.orthog_gamma,
                                             model=self.x2c)
        else:
            orth_loss = -1
        

        # composite loss (concept + task + other losses)
        combined_loss = self._calculate_loss_optimization(concept_loss, 
                                                          task_loss, 
                                                          orth_loss,
                                                          eval=False)

        # log accuracy + loss metrics
        self.log_metrics((c_pred,c,y_pred,y))

        return combined_loss
    
    def on_train_epoch_end(self):
        # step schedulers

        # log epsilon at end of each epoch
        if self.config.model.train_private:
            budget = self._get_epsilon()
            self.log('privacy_budget', budget)

        sch1, sch2 = self.lr_schedulers()
        sch1.step(self.trainer.callback_metrics["combined_loss"]) # x2c_lr_scheduler
        sch2.step(self.trainer.callback_metrics["combined_loss"]) # c2y_lr_scheduler

    def log_metrics(self, accs):
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

    # DP step, setup opacus components (optimizer, privacy_engine, loss)
    def _setup_dp(self):
        # https://discuss.pytorch.org/t/pytorch-lightning-support/113507/7
        
        # Validate and fix model for Opacus compatibility
        if not ModuleValidator.is_valid(self.x2c):
            print("Model not compatible with Opacus, attempting to fix...")
            self.x2c = ModuleValidator.fix(self.x2c.parameters())
            print("Model fixed for Opacus compatibility")
        """
        steps for optimizer(s)
        parts to model:
        1. x2c - has a frozen backbone (gradients are false), and a trainable head
            -> can you train all of self.x2c this way? No, it is the norm to only set head as trainable by optimizer
        2. c2y - a linear layer. all of it is trainable
        """
        optimizer = torch.optim.SGD(self.x2c.model[0][1].parameters(),
                            lr=self.config.model.model_args.lr,
                            momentum=0.9)
        privacy_engine = None
        try:
            # DPSGD (no gradient alteration)
            privacy_engine = PrivacyEngine()
            _, optimizer_gc, _, _ = privacy_engine.make_private_with_epsilon(
                module=self.x2c.model[0][1],
                optimizer=optimizer,
                data_loader=self.train_dl,
                target_delta=self.config.model.privacy_args.delta,
                target_epsilon=self.config.model.privacy_args.epsilon,
                epochs=self.config.data.epochs,
                max_grad_norm=self.config.model.privacy_args.max_grad_norm,
                grad_sample_mode="ghost", # reduces memory overhead in grad calc step
                criterion=self.loss_before_dp
            )
            print("Successfully initialized PrivacyEngine...")
        except Exception as e:
            print(f"Failed to initialize PrivacyEngine: {e}")
            raise e
        
        # update arguments passed to fit
        return optimizer_gc, None, privacy_engine # optimizer, loss, privacy_engine
    
    def setup(self, stage):
        # https://discuss.pytorch.org/t/pytorch-lightning-support/113507/7
        # run DP setup at beginning of test/train/validation
        if self.config.model.train_private:
            self.loss_concept = self.loss_before_dp
            self.x2c_optimizer, _, self.privacy_engine = self._setup_dp()
        else:
            self.x2c_optimizer = torch.optim.SGD(self.x2c.parameters(),
                            lr=self.config.model.model_args.lr,
                            momentum=0.9)
            self.loss_concept = self.loss_before_dp
    
    # DP step, log privacy budget
    def _get_epsilon(self):
        try:
            budget = self.privacy_engine.get_epsilon(self.config.model.privacy_args.delta)
            return budget
        except:
            return -1
        
    def validation_step(self, batch, batch_idx):
        #  we are given X,Y,C
        x,y,c = batch
        y = y.long()

        # pred
        c_pred, y_pred = self(x)

        # loss
        concept_loss = self.loss_concept(c_pred, c)
        task_loss = f.cross_entropy(y_pred, y)
        orth_loss = None
        if self.config.model.train_private and self.config.model.privacy_args.use_orthog_loss:
            orth_loss = ot_loss.OrthogonalLoss()(gamma=self.config.model.privacy_args.orthog_gamma,
                                             model=self.x2c)
        else:
            orth_loss = -1

        # composite loss (concept + task + other losses)
        combined_loss = self._calculate_loss_optimization(concept_loss, 
                                                          task_loss, 
                                                          orth_loss,
                                                          eval=True)
        
        # log accuracy + loss metrics
        self.log_metrics((c_pred,c,y_pred,y))

        return combined_loss
    
    def on_validation_epoch_end(self):
        # keep self.config info in a readme
        version_path = f"{self.trainer.default_root_dir}/lightning_logs/version_{self.logger.version}"
        if not os.path.exists(f'{version_path}/readme.md'):
            with open(f'{version_path}/readme.md', 'w') as f:
                f.write(f"config: {self.config}")
                # f.write(f"model: {self.config.model.model_name}\n")
                # if self.config.model.train_private:
                #     f.write(f"epsilon: {self.config.model.privacy_args.epsilon}\n")
                #     f.write(f"delta: {self.config.model.privacy_args.delta}\n")
                #     f.write(f"max grad norm: {self.config.model.privacy_args.max_grad_norm}\n")
                #     if self.config.model.privacy_args.use_orthog_loss:
                #         f.write(f"use orthogonal loss: {self.config.model.privacy_args.use_orthog_loss}\n")
                #     if self.config.model.privacy_args.use_orthog_clipping:
                #         f.write(f"use orthogonal clipping: {self.config.model.privacy_args.use_orthog_clipping}\n")

    # create readme.md with run config info, save ckpt every 10 epochs 
    def on_validation_end(self):
        # manual optimization seems to break logging, save ckpt manually
        version_path = f"{self.trainer.default_root_dir}/lightning_logs/version_{self.logger.version}"
        checkpoint_path = f"{version_path}/checkpoints/epoch-{self.current_epoch}.ckpt"

        # save every 10 epochs
        if (self.current_epoch % 10 == 0) or (self.current_epoch == self.trainer.max_epochs-1):
            self.trainer.save_checkpoint(checkpoint_path)

    def test_step(self, batch, batch_idx):
        x,y,c = batch # always in form x,y,c
        y = y.long()

        # pred
        c_pred, y_pred = self(x)

        # loss + accuracy
        concept_loss = self.loss_concept(c_pred, c)
        task_loss = f.cross_entropy(y_pred, y)
        orth_loss = None
        if self.config.model.train_private and self.config.model.privacy_args.use_orthog_loss:
            orth_loss = ot_loss.OrthogonalLoss()(gamma=self.config.model.privacy_args.orthog_gamma,
                                             model=self.x2c)
        else:
            orth_loss = -1
        
        # log composite loss (concept + task + other losses)
        combined_loss = self._calculate_loss_optimization(concept_loss, 
                                                          task_loss, 
                                                          orth_loss,
                                                          eval=True)
        # log accuracy + loss metrics
        # bootstrap metrics
        self.task_auc_bs.update(y_pred, y)
        self.concept_auc_bs.update(c_pred, c.long())
        self.task_bootstrapper.update(y_pred, y)
        self.concept_bootstrapper.update(c_pred, c.long())

        return combined_loss

    def on_test_epoch_end(self):
        # get upper/lower bound
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

    def _calculate_loss_optimization(self, concept_loss, task_loss,orth_loss, eval=False):  
        combined_loss = None
        if self.config.model.train_private and self.config.model.privacy_args.use_orthog_loss:     
            combined_loss = (self.concept_loss_weight * concept_loss.mean()) + (self.task_loss_weight * task_loss) + orth_loss
            self.log('orthog_loss', orth_loss)
            
        else:
            combined_loss = (self.concept_loss_weight * concept_loss.mean()) + (self.task_loss_weight * task_loss)
        
        # log loss
        self.log('concept_loss', concept_loss.mean())
        self.log('task_loss', task_loss)
        self.log('combined_loss', combined_loss)
        
        if not eval:
            self.x2c_optimizer.zero_grad()
            self.c2y_optimizer.zero_grad()

            self.manual_backward(combined_loss)

            self.x2c_optimizer.step()
            self.c2y_optimizer.step()

        # note: schedulers are not called automatically
        # called manually in `on_train_epoch_end`
        return combined_loss
        
    def configure_optimizers(self):
        """
        cbm uses two optimizers.
       
        self.x2c_optimizer is a DP optimizer for training the head (last BasicBlock + output layer)
        of x2c model

        self.c2y_optimizer is a non DP optimizer for training the linear layer
        """
        # x2c_optimizer defined in `setup()`

        self.c2y_optimizer = torch.optim.SGD(self.c2y.parameters(),
                                        lr=self.config.model.model_args.lr,
                                        momentum=0.9)
        x2c_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.x2c_optimizer,
            verbose=True,
        )
        c2y_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.c2y_optimizer,
            verbose=True,
        )

        return [self.x2c_optimizer, self.c2y_optimizer], [x2c_lr_scheduler,c2y_lr_scheduler]
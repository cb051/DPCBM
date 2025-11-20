import torch
import matplotlib.pyplot as plt
from torch import nn
import torch
import torch.nn.functional as f

import lightning as L
import torchmetrics
from torchmetrics.wrappers import BootStrapper

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import sys

import utils.resnet34 as res34
from utils.x2c import X2C
from utils.c2y import C2Y

from torchmetrics.wrappers import BootStrapper
from torchmetrics.classification import Accuracy, MultilabelAccuracy
import os
from omegaconf import DictConfig, OmegaConf
import hydra

import datasets.mnist_dataset as mnist_ds
import datasets.awa2_dataset as awa2_ds
import datasets.cub200_dataset as cub200_ds


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
  
    def orthogonal_regularization(self, y_pred, y):
        # look at orthogonality in convolutional layers
        # return penalty term
        fro_norms = []
        for name, params in self.named_parameters():
            if ".conv" in name:
                # calculate dot product AA^T
                I = torch.eye(params.shape[-1])
                I = I.repeat(params.shape[0],params.shape[1],1,1) 
                mat = torch.matmul(params, torch.swapaxes(params,-1,-2)).cpu() - I.cpu() # ||AA^T -I||^2_F
                sq_norm = torch.linalg.norm(mat**2) # squared frobenius norm
                fro_norms.append(sq_norm)
        
        loss = torch.mean(torch.stack(fro_norms))

        return 0.1 * loss

    def training_step(self, batch, batch_idx):
        #  we are given X,Y,C
        x,y,c = batch
        y = y.long()

        # pred
        c_pred, y_pred = self(x)

        # loss + accuracy + epsilon
        x2c_loss = self.loss_concept(c_pred, c)
        c2y_loss = f.cross_entropy(y_pred, y)
        orth_loss = self.orthogonal_regularization(y_pred,y)
        

        # composite loss (concept + task + other losses)
        combined_loss = self._calculate_loss_optimization(x2c_loss, c2y_loss, orth_loss,eval=False)

        # log accuracy + loss metrics
        self.log_metrics((c_pred,c,y_pred,y),
                         log_only=True,
                         update_only=False,
                         compute_only=False,
                         reset_metric=False)

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

        #  reset metrics
        self.c_accuracy.reset()
        self.y_accuracy.reset()
        self.concept_auc.reset()
        self.task_auc.reset()
        self.concept_f1.reset()
        self.task_f1.reset()

    def log_metrics(self,accs,log_only=False, update_only=False, compute_only=False,reset_metric=False):
        
        if log_only and (accs is not None):
            c_pred,c,y_pred,y = accs
            c_accuracy = self.c_accuracy(c_pred,c)
            y_accuracy = self.y_accuracy(y_pred, y)
            concept_auc = self.concept_auc(c_pred,c.long())
            task_auc = self.task_auc(y_pred, y)
            concept_f1 = self.concept_f1(c_pred,c)
            task_f1 = self.task_f1(y_pred, y)
        elif update_only and (accs is not None):
            self.concept_bootstrapper.update(c_pred, c)
            self.task_bootstrapper.update(y_pred, y)
            self.concept_auc_bs.update(c_pred,c.long())
            self.task_auc_bs.update(y_pred, y)

            self.c_accuracy.update(c_pred,c)
            self.y_accuracy.update(y_pred, y)
            self.concept_auc.update(c_pred,c.long())
            self.task_auc.update(y_pred, y)
            self.concept_f1.update(c_pred,c)
            self.task_f1.update(y_pred, y)
        elif compute_only:
            c_accuracy = self.c_accuracy.compute()
            y_accuracy = self.y_accuracy.compute()
            concept_auc = self.concept_auc.compute()
            task_auc = self.task_auc.compute()
            concept_f1 = self.concept_f1.compute()
            task_f1 = self.task_f1.compute()
        # log metrics
        self.log('task_accuracy (y)', y_accuracy)
        self.log('concept_accuracy (c)', c_accuracy)
        self.log('concept_auc', concept_auc)
        self.log('task_auc', task_auc)
        self.log('concept_f1', concept_f1)
        self.log('task_f1', task_f1)

        if reset_metric:
            self.c_accuracy.reset()
            self.y_accuracy.reset()
            self.concept_auc.reset()
            self.task_auc.reset()
            self.concept_f1.reset()
            self.task_f1.reset()



    # DP step, setup opacus components (optimizer, privacy_engine, loss)
    def _setup_dp(self):
        # https://discuss.pytorch.org/t/pytorch-lightning-support/113507/7
        
        # Validate and fix model for Opacus compatibility
        if not ModuleValidator.is_valid(self.x2c.model[0][1]):
            print("Model not compatible with Opacus, attempting to fix...")
            self.x2c.model[0][1] = ModuleValidator.fix(self.x2c.model[0][1])
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
            _, optimizer_gc, loss_gc, _ = privacy_engine.make_private_with_epsilon(
                module=self.x2c.model[0][1],
                optimizer=optimizer,
                data_loader=self.train_dl,
                target_delta=self.config.model.privacy_args.delta,
                target_epsilon=self.config.model.privacy_args.epsilon,
                epochs=self.config.data.epochs,
                max_grad_norm=self.config.model.privacy_args.max_grad_norm,
                grad_sample_mode="ghost", # reduces memory overhead in grad calc step
                
            )
            print("Successfully initialized PrivacyEngine...")
        except Exception as e:
            print(f"Failed to initialize PrivacyEngine: {e}")
            raise e
        
        # update arguments passed to fit
        return optimizer_gc, loss_gc, privacy_engine
    
    def setup(self, stage):
        # https://discuss.pytorch.org/t/pytorch-lightning-support/113507/7
        # run DP setup at beginning of test/train/validation
        if self.config.model.train_private:
            self.x2c_optimizer, self.loss_concept, self.privacy_engine = self._setup_dp()
        else:
            self.x2c_optimizer = torch.optim.SGD(self.x2c.model[0][1].parameters(),
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
        x2c_loss = self.loss_concept(c_pred, c)
        c2y_loss = f.cross_entropy(y_pred, y)
        orth_loss = self.orthogonal_regularization(y_pred,y)

        # composite loss (concept + task + other losses)
        combined_loss = self._calculate_loss_optimization(x2c_loss, 
                                                          c2y_loss, 
                                                          orth_loss,
                                                          eval=True)
        
        # log accuracy + loss metrics
        self.log_metrics((c_pred,c,y_pred,y),
                         log_only=True,
                         update_only=False,
                         compute_only=False,
                         reset_metric=False)

        return combined_loss
    
    def on_validation_epoch_end(self):

        # compute,log, and reset accuracy + loss metrics
        self.log_metrics(None,
                         update_only=False,
                         compute_only=True,
                         reset_metric=True)
        
        # keep self.config info in a readme
        version_path = f"{self.trainer.default_root_dir}/lightning_logs/version_{self.logger.version}"
        if not os.path.exists(f'{version_path}/readme.md'):
            with open(f'{version_path}/readme.md', 'w') as f:
                f.write(f"model: {self.config.model.model_name}")
                if self.config.model.train_private:
                    f.write(f"epsilon: {self.config.model.privacy_args.epsilon}")

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
        c_pred, y_pred = self._predict(x)

        # loss + accuracy
        x2c_loss = self.loss_concept(c_pred, c)
        c2y_loss = f.cross_entropy(y_pred, y)

        if config.model.model_args.use_orthog_loss:
            orth_loss = self.orthogonal_regularization(y_pred,y)
            combined_loss = (self.concept_loss_weight * x2c_loss.item()) + (self.task_loss_weight * c2y_loss) + orth_loss
        else:
            combined_loss = (self.concept_loss_weight * x2c_loss.item()) + (self.task_loss_weight * c2y_loss)
        
        # log composite loss (concept + task + other losses)
        combined_loss = self._calculate_loss_optimization(x2c_loss, 
                                                          c2y_loss, 
                                                          orth_loss,
                                                          eval=True)
        # log accuracy + loss metrics
        self.log_metrics((c_pred,c,y_pred,y),
                         log_only=False,
                         update_only=True,
                         compute_only=False,
                         reset_metric=False)

        return combined_loss

    def on_test_epoch_end(self):
        # compute, log, and reset accuracy + loss metrics
        self.log_metrics(None,
                         log_only=False,
                         update_only=False,
                         compute_only=True,
                         reset_metric=True)

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
        self.log('C acc', (c_bs['mean']*100), prog_bar=True)
        self.log('Y acc', (y_bs['mean']*100), prog_bar=True)
        print(f"=== 95% CI concept acc: {c_bs['quantile'][0] * 100} {c_bs['quantile'][1] * 100} ===")
        print(f"=== 95% CI task acc: {y_bs['quantile'][0] * 100} {y_bs['quantile'][1] * 100} ===")
        print(f"=== 95% CI concept AUC: mean: {c_auc_bs['mean']} upper: {c_auc_bs['quantile'][0] * 100} lower: {c_auc_bs['quantile'][1] * 100} ===")
        print(f"=== 95% CI task AUC: mean: {y_auc_bs['mean']} upper: {y_auc_bs['quantile'][0] * 100} lower: {y_auc_bs['quantile'][1] * 100} ===")

    def _calculate_loss_optimization(self, concept_loss, task_loss,orth_loss, eval=False):  
        combined_loss = None
        if self.config.model.train_private and self.config.model.model_args.use_orthog_loss:      
            combined_loss = (self.concept_loss_weight * concept_loss.item()) + (self.task_loss_weight * task_loss) + orth_loss
            self.log('orthog_loss', orth_loss)
        else:
            combined_loss = (self.concept_loss_weight * concept_loss.item()) + (self.task_loss_weight * task_loss)
        
        # log loss
        self.log('concept_loss', concept_loss.item())
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


#=======================================
#  main ================================
#=======================================
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    print(f"CHECK CONFIG: {OmegaConf.to_yaml(config)}")
    # set dataloaders
    if config.data.dataset_name == 'mnist':
        train_dl, val_dl = mnist_ds.MNIST().get_dataloaders(batch_size=config.batch_size, NDW=config.num_workers)
    elif config.data.dataset_name == 'cub200':
        train_dl, val_dl = cub200_ds.CUB().get_dataloaders(batch_size=config.batch_size, NDW=config.num_workers)
    elif config.data.dataset_name == 'awa2':
        train_dl, val_dl = awa2_ds.get_dataloaders(batch_size=config.batch_size, NDW=config.num_workers)

    # train cbm + private variants
    cbm = CBM(
                train_dl=train_dl,
                config=config,
                )
    # setup model (name local logger paths according to experiment)
    if config.model.train_private:
        logger_path = f"{config.model.logging.local_path}/{config.model.logging.ckpt_path_name}_" \
            f"{config.data.dataset_name}_e{config.model.privacy_args.epsilon}"
    elif config.model.train_private and config.model.train_private.use_orthog_loss:
         logger_path = f"{config.model.logging.local_path}/{config.model.logging.ckpt_path_name}_" \
            f"{config.data.dataset_name}_e{config.model.privacy_args.epsilon}_orthog_loss"
    elif config.model.train_private and config.model.train_private.use_orthog_clipping:
        logger_path = f"{config.model.logging.local_path}/{config.model.logging.ckpt_path_name}_" \
            f"{config.data.dataset_name}_e{config.model.privacy_args.epsilon}_orthog_clipping"
    else:
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
    

    trainer = L.Trainer(accelerator=config.accelerator,
                        devices=config.devices,
                        max_epochs=1 if config.one_epoch_run else config.data.epochs, 
                        default_root_dir=logger_path if config.model.logging.log_local else None,
                        callbacks=[lr_monitor,save_every_10_epochs],
                        logger=logger)

    if config.train:
        # train model
        trainer.fit(cbm,
                    train_dl,
                    val_dl,
                    )
    elif config.test:
        trainer.test(config.ckpt,val_dl)

if __name__=='__main__':
    main()
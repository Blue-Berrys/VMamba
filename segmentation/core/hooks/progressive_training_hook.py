"""
MMSegmentation渐进式训练Hook
==============================

实现三阶段训练策略的Hook，自动切换训练阶段

集成路径: /home/xjx/CodeProject/PycharmProject/VMamba/segmentation/core/hooks/

使用方法:
在配置文件中添加:
```python
custom_hooks = [
    dict(
        type='ProgressiveTrainingHook',
        stage1_iters=20000,
        stage2_iters=10000,
        stage3_iters=10000
    )
]
```

作者: AgentLaboratory
日期: 2026-02-19
"""

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS


@HOOKS.register_module()
class ProgressiveTrainingHook(Hook):
    """
    Progressively freeze/unfreeze backbone streams.

    Stage 0: train global_stream + heads only
    Stage 1: train local_stream + fusion + heads only
    Stage 2: train all parameters jointly

    Only controls requires_grad; LR schedule is left to param_scheduler.
    """

    def __init__(
        self,
        stage1_iters: int = 20000,
        stage2_iters: int = 10000,
        stage3_iters: int = 10000,
        log_stage_switch: bool = True
    ):
        self.stage1_iters = stage1_iters
        self.stage2_iters = stage2_iters
        self.stage3_iters = stage3_iters
        self.log_stage_switch = log_stage_switch
        self.current_stage = -1
        self.stage_names = ['global_stream', 'local_stream', 'joint']

    def get_current_stage(self, iter_num: int) -> int:
        if iter_num < self.stage1_iters:
            return 0
        elif iter_num < self.stage1_iters + self.stage2_iters:
            return 1
        else:
            return 2

    def before_train_iter(self, runner, batch_idx, data_batch):
        new_stage = self.get_current_stage(runner.iter)
        if new_stage != self.current_stage:
            self._switch_stage(runner, new_stage)

    def _switch_stage(self, runner, new_stage: int):
        self.current_stage = new_stage

        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        backbone = model.backbone if hasattr(model, 'backbone') else model
        if hasattr(backbone, 'set_training_stage'):
            backbone.set_training_stage(new_stage)

        # heads are always trainable
        for name in ['decode_head', 'auxiliary_head']:
            head = getattr(model, name, None)
            if head is not None:
                for param in head.parameters():
                    param.requires_grad = True

        if self.log_stage_switch:
            runner.logger.info('=' * 50)
            runner.logger.info(
                f'Stage switch -> stage {new_stage}: {self.stage_names[new_stage]}'
            )
            runner.logger.info('=' * 50)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            runner.logger.info(
                f'Trainable: {trainable/1e6:.2f}M / Total: {total/1e6:.2f}M'
            )


@HOOKS.register_module()
class AuxiliaryLossHook(Hook):
    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        pass


__all__ = ['ProgressiveTrainingHook', 'AuxiliaryLossHook']

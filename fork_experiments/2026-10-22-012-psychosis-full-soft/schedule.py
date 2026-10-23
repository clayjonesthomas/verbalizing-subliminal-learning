"""Absolute validation-NLL plateau rule, independent of best checkpoint selection."""
from dataclasses import dataclass,asdict
@dataclass
class Plateau:
    warmup_steps:int=100
    patience:int=5
    min_delta:float=.001
    factor:float=.5
    max_reductions:int=2
    reference:float|None=None
    bad_evaluations:int=0
    reductions:int=0
    stopped:bool=False
    def rate(self,base,step):
        return base*min(1.,step/self.warmup_steps)*self.factor**self.reductions
    def observe(self,step,nll):
        if step<self.warmup_steps:return 'warmup'
        if self.reference is None or nll<=self.reference-self.min_delta:
            self.reference=nll;self.bad_evaluations=0;return 'improved'
        self.bad_evaluations+=1
        if self.bad_evaluations<self.patience:return 'continue'
        if self.reductions<self.max_reductions:
            self.reductions+=1;self.bad_evaluations=0;return 'halve_learning_rate'
        self.stopped=True;return 'early_stop'
    def state(self):return asdict(self)

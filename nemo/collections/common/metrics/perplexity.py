# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from pytorch_lightning.metrics import Metric
from torch.distributions.categorical import Categorical

__all__ = ['Perplexity']


class Perplexity(Metric):
    """
    This class computes mean perplexity of distributions in the last dimension of inputs. It is a wrapper around
    :doc:`torch.distributions.Categorical.perplexity<pytorch:distributions>` method. You have to provide either 
    ``probs`` or ``logits`` to the :meth:`update` method. The class computes perplexities for distributions passed to 
    :meth:`update` method in ``probs`` or ``logits`` arguments and averages the perplexities. Reducing results between
    all workers is done via SUM operations.
    See :doc:`PyTorch Lightning Metrics<pytorch-lightning:metrics>` for the metric usage instructions.
    Args:
        compute_on_step:
            Forward only calls ``update()`` and returns ``None`` if this is set to ``False``. default: ``True``
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step.
        process_group:
            Specify the process group on which synchronization is called. default: ``None`` (which selects the entire
                world)
        validate_args:
            If ``True`` values of :meth:`update` method parameters are checked. ``logits`` has to not contain NaNs and
            ``probs`` last dim has to be valid probability distribution.
    """

    def __init__(self, compute_on_step=True, dist_sync_on_step=False, process_group=None, validate_args=True):
        super().__init__(
            compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step, process_group=process_group
        )
        self.validate_args = validate_args
        self.add_state('perplexities_sum', torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx='sum')
        # Total number of distributions seen since last reset
        self.add_state('num_distributions', torch.tensor(0, dtype=torch.int64), dist_reduce_fx='sum')

    def update(self, probs=None, logits=None):
        """
        Updates :attr:`perplexities_sum` and :attr:`num_distributions`.
        Args:
            probs: A ``torch.Tensor`` which innermost dimension is valid probability distribution.
            logits: A ``torch.Tensor`` without NaNs.
        """
        d = Categorical(probs, logits, validate_args=self.validate_args)
        ppl = d.perplexity()
        self.num_distributions += ppl.numel()
        self.perplexities_sum += ppl.sum()

    def compute(self):
        """
        Returns perplexity across all workers and resets to 0 :attr:`perplexities_sum` and :attr:`num_distributions`.
        """
        if self.num_distributions.eq(0):
            return None
        return self.perplexities_sum / self.num_distributions

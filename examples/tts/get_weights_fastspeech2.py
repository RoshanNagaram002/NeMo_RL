# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import pytorch_lightning as pl
import torch

from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.models import FastSpeech2Model
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager


if __name__ == '__main__':
    # pre_model = FastSpeech2Model.from_pretrained("tts_en_fastspeech2")
    # torch.save(pre_model.state_dict(), 'pretrained_fastspeech2')

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = FastSpeech2Model(cfg=cfg.model, trainer=trainer)
    model.load_state_dict(torch.load("pretrained_fastspeech2"))
    


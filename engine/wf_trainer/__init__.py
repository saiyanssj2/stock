# -*- coding: utf-8 -*-
"""
Walk-Forward Hybrid SL+RL Trainer.

Kiến trúc 2 giai đoạn:
- Stage 1 (Supervised): Pretrain TCN+Attention để nhận diện patterns
- Stage 2 (RL): Fine-tune với PPO để tối ưu trading decisions

Cycle: update data → SL pretrain → RL finetune → backtest → report → lặp lại
"""

from engine.wf_trainer.walk_forward import WalkForwardCycle, run_one_cycle

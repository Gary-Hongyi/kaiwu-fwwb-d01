#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

try:
    from kaiwudrl.common.utils.train_test_utils import run_train_test
except ModuleNotFoundError:
    run_train_test = None

# To run the train_test, you must modify the algorithm name here. It must be one of ppo, diy.
# Simply modify the value of the algorithm_name variable.
# 运行train_test前必须修改这里的算法名字, 必须是ppo、diy里的一个, 修改algorithm_name的值即可
algorithm_name_list = ["ppo", "diy"]
algorithm_name = "ppo"


if __name__ == "__main__":
    if run_train_test is not None:
        run_train_test(
            algorithm_name=algorithm_name,
            algorithm_name_list=algorithm_name_list,
            env_vars={
                "replay_buffer_capacity": "10",
                "preload_ratio": "1.0",
                "train_batch_size": "2",
                "dump_model_freq": "1",
            },
        )
    else:
        import numpy as np
        from agent_diy.agent import Agent
        from agent_diy.conf.conf import Args, Config
        from agent_diy.feature.definition import ObsData, SampleData

        agent = Agent(device="cpu")
        assert agent.init_config({"my_heroes": [112, 133]}) == {112: 80115, 133: 80115}
        obs_data = ObsData(
            feature=np.zeros(Args.DIM_ALL, dtype=np.float32),
            legal_action=np.ones(sum(Config.LEGAL_ACTION_SIZE_LIST), dtype=np.float32),
            lstm_cell=agent.lstm_cell,
            lstm_hidden=agent.lstm_hidden,
        )
        act_data = agent._model_inference([obs_data])[0]
        assert len(act_data.action) == len(Config.LABEL_SIZE_LIST)
        sample = SampleData(np.zeros(Config.SAMPLE_DIM, dtype=np.float32))
        assert sample.sample.shape[0] == Config.SAMPLE_DIM
        print("Local smoke train_test fallback passed: agent import, init_config, inference, SampleData")

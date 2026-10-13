#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Minimal learner for Gorge Chase DIY agent.
"""


import os
import time


class Algorithm:
    def __init__(self, model, optimizer=None, scheduler=None, device=None, logger=None, monitor=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.monitor = monitor
        self.train_step = 0
        self.last_report_monitor_time = 0

    def learn(self, list_sample_data):
        self.train_step += 1
        if not list_sample_data:
            return

        avg_reward = sum(self._to_scalar(sample.reward) for sample in list_sample_data) / len(list_sample_data)
        now = time.time()
        if self.logger and now - self.last_report_monitor_time >= 60:
            self.logger.info(
                f"[diy-train] step:{self.train_step} sample_count:{len(list_sample_data)} avg_reward:{avg_reward:.4f}"
            )
            if self.monitor:
                self.monitor.put_data({os.getpid(): {"avg_reward": round(avg_reward, 4)}})
            self.last_report_monitor_time = now

    def _to_scalar(self, value):
        if hasattr(value, "item"):
            try:
                return float(value.item())
            except Exception:
                pass
        if isinstance(value, (list, tuple)):
            return float(value[0]) if value else 0.0
        try:
            return float(value[0])
        except Exception:
            return float(value)

    def _to_scalar(self, value):
        if hasattr(value, "item"):
            try:
                return float(value.item())
            except Exception:
                pass
        if isinstance(value, (list, tuple)):
            return float(value[0]) if value else 0.0
        try:
            return float(value[0])
        except Exception:
            return float(value)

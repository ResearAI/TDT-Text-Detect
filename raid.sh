#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

data_path=./benchmark/raid
result_path=./exp_raid
mkdir -p $result_path

datasets='raid.dev,raid.test,nonnative.test'
detectors='tdt'

python scripts/delegate_detector.py --data_path $data_path --result_path $result_path \
              --datasets $datasets --detectors $detectors

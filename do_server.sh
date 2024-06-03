#!/bin/bash

# 重建会话：
# tmux new -s xj
# 按 ctrl+B 后 输入冒号： 之后输入 set -g mouse on 可以开启鼠标上下滚动
# conda activate xj_rlbench

# 正常使用：
# tmux a -t xj
# . do_server.sh

git checkout .
git pull

bash train_eval.sh
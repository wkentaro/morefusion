#!/bin/bash

logtable --hide class_names multi_node gpu max_epoch voxelization '.*main/add.*' '.*main/loss.*' freeze_until githash hostname debug 'validation/main/auc.*(min)' '.*add_r.*' 'validation/main/auc/add/.*' 'validation/main/auc/add_s/.*' nocall_evaluation_before_training timestamp out seed $*

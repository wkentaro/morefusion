#!/bin/bash

logtable --hide class_names multi_node gpu max_epoch voxelization '.*main/add.*' '.*main/loss.*' freeze_until githash hostname debug 'validation/main/auc.*(min)' 'validation/main/auc/add_rotation/.*' 'validation/main/auc/add/.*' nocall_evaluation_before_training timestamp out seed $*

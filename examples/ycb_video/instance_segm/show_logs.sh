#!/bin/bash

logtable --filter 'validation/main.*(min)' 'validation/main/mar/.*' 'main/loss.*(max)' 'main/loss/.*' 'lr.*' 'validation/main/map/iou=0.50/.*' 'validation/main/map/iou=0.75/.*'

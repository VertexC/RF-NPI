"""
main.py
"""
import argparse
import sys 
import os
import shutil
# FIXME: make import task specific?
def import_task():
    pass
from tasks.addition.eval import model_eval
from tasks.addition.train import model_train
from tasks.addition.env.generate import generate_data

# FIXME: use argparse?
task = 'addition'
generate = 'True'
num_train = 1000
num_test = 1000
do_train = False
do_eval = False
num_epochs = 5



if task == 'addition':
    if generate:
        generate_data('train', num_train, debug=True)
        generate_data('test', num_test)
    if do_train:
        model_train(num_epochs)
    if do_eval:
        model_eval()








#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   util.py
@Time    :   2020/02/01 17:31:28
@Version :   1.0
@Describtion:  other functions: make log folder, save and load model, 

'''

# here put the import lib
import os
import torch
import pickle
import logging
import time

from model import build_model


def load(check_point_name,model_storage_path,ARGS):
    '''Load model from .pt file.  
    Args:  
        check_point_name(str):   Filename to store best model
        model_storage_path(str): Folder path to store best model
    Returns:
        model(nn.Module):        Loaded model
    '''
    save_path = os.path.join(model_storage_path+str(ARGS.gamename), check_point_name)
    model = build_model(ARGS)
    model.load_state_dict(torch.load(save_path))
    return model

def save(model_best, checkpoint_name,model_storage_path,gen):
    '''save model into .pt file  
    Args:  
        check_point_name(str):   filename to store best model  
        model_storage_path(str): folder path to store best model  
    '''
    save_path = os.path.join(model_storage_path, checkpoint_name+str(gen)+'.pt')
    torch.save(model_best.state_dict(), save_path)
    return save_path

def setup_logging(logger,folder_path,filename,txtlog=True,scrlog=False):
    """Create and init logger. 
    Reset hander.   
    Args:
        logger:          logger class
        logfolder_path:  "/log/"
        filename:        "Alien-phi-0.001-mu-14.txt"
        txtlog(bool):    If True, print log into file
        scrlog(bool):    If True, print log into screen
    Return:
        logger:           
    """
    logfile = os.path.join(folder_path, filename)
    logging.basicConfig(
        level = logging.INFO,
        format ='%(asctime)s - %(levelname)s - %(message)s',
        filename = logfile,
        filemode = 'a'
    )
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    if logger.hasHandlers() is True:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    if txtlog is True:
        # handler = logging.FileHandler(logfile,mode='a',encoding='utf-8')
        # handler.setLevel(logging.INFO)
        # handler.setFormatter(formatter) 
        # logger.addHandler(handler) 
        logger.info("Logger initialised.") 
    if scrlog is True:
        console_handler = logging.StreamHandler() 
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter) 
        logger.addHandler(console_handler)

    return logger

def mk_folder(LogFolder,args):
    '''make folder to save log'''
    timenow = time.localtime(time.time())
    indx = 1
    GameFolder = os.path.join(LogFolder,gamename)
    if gamename not in os.listdir(LogFolder):
        os.mkdir(GameFolder)
    logfolder = str(timenow.tm_year)+'-'+str(timenow.tm_mon)+'-'+str(timenow.tm_mday)+'-'+str(indx)
    while logfolder in os.listdir(GameFolder):
        indx += 1
        logfolder = str(timenow.tm_year)+'-'+str(timenow.tm_mon)+'-'+str(timenow.tm_mday)+'-'+str(indx)
    logfolder_path = os.path.join(GameFolder, logfolder)
    os.mkdir(logfolder_path)
    print("make folder:", logfolder_path)
    return logfolder_path

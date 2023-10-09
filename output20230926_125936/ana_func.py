#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
from nilearn import surface
import logging
import colorlog
import datetime

def calDiceROI(rest1,rest2,totalArea=106): 
    parcels = totalArea
    sumdice = 0

    if len(rest1.shape) > 1:
        rest1 = np.argmax(rest1, axis=1)
        rest2 = np.argmax(rest2, axis=1)

        for parcel in range(parcels):
            rest1Parcel = np.argwhere(rest1 == parcel)
            rest2Parcel = np.argwhere(rest2 == parcel)
            if len(rest1Parcel)+len(rest2Parcel) == 0:
                continue
                #return -1
            parcelDice = 2*len(set(rest1Parcel.flatten()) & set(rest2Parcel.flatten()))/(len(rest1Parcel)+len(rest2Parcel))
            sumdice +=parcelDice

        return sumdice/parcels
    else:
        # label_set = sorted(set(rest1))
        # dice = []
        # for label_ix in label_set:
            # rest1Parcel = np.argwhere(rest1 == label_ix)
            # rest2Parcel = np.argwhere(rest2 == label_ix)
            # dice.append(2*len(set(rest1Parcel.flatten()) & set(rest2Parcel.flatten()))/(len(rest1Parcel)+len(rest2Parcel)))
        # return dice
        dice = sum(rest1==rest2) / (len(rest2)+0.0001)
        return dice

def diceStats(atlasPath, test_dict, hemi):

    subnames = sorted(list(test_dict.keys()))
    subnum = len(subnames)
    intraSubject = []
    interSubject = []
    group_df = pd.DataFrame(columns=["sub_name", "sess_num", "dice"])
    intra_df = pd.DataFrame(columns=["sub_name", "sess_num", "sess_other", "dice"])
    inter_df = pd.DataFrame(columns=["sub_name", "sess_num", "sub_other", "sess_other", "dice"])
    group_label = surface.load_surf_data("/nfs2/users/zj/Brainnetome/fsaverage.{}.BN_Atlas.10k_fsaverage.label.gii".format(hemi))
    file_general = ".indi.{}.label.gii".format(hemi)
    for i in range(subnum):

        # if subnames[i] in ["sub-25638","sub-25632_wrong"]:
            # continue

        sessnames = sorted(list(test_dict[subnames[i]]))
        sessnum = len(sessnames)
        for sess_ix in range(sessnum):
            # rltPath = f"{atlasPath}/{subnames[i]}_{sessnames[sess_ix]}_{hemi}_10k.label.gii"

            rltPath = "{}/{}_{}".format(atlasPath, subnames[i], sessnames[sess_ix])+file_general
            if not os.path.isfile(rltPath):
                continue
            rlt = surface.load_surf_data(rltPath)
            rlt_group = calDiceROI(rlt,group_label)
            group_df = group_df.append({"sub_name":subnames[i], "sess_num":sessnames[sess_ix], "dice":rlt_group},ignore_index=True)
            for sess_ix1 in range(sess_ix+1,sessnum):
                # rltPath1 = f"{atlasPath}/{subnames[i]}_{sessnames[sess_ix1]}_{hemi}_10k.label.gii"
                rltPath1 = "{}/{}_{}".format(atlasPath, subnames[i], sessnames[sess_ix1])+file_general
                if not os.path.isfile(rltPath1):
                    continue
                rlt1 = surface.load_surf_data(rltPath1)
                rlt_rlt1 = calDiceROI(rlt,rlt1)
                print("{} {} and {} intraSubject is: {}".format(subnames[i], sessnames[sess_ix], sessnames[sess_ix1], rlt_rlt1))
                intra_df = intra_df.append({"sub_name":subnames[i], "sess_num":sessnames[sess_ix],"sess_other": sessnames[sess_ix1], "dice":rlt_rlt1},ignore_index=True)
                intraSubject.append(rlt_rlt1)
            for j in range(i+1,subnum):
                # if subnames[j] in ["sub-25638","sub-25632_wrong"]:
                    # continue

                sessnames_other = sorted(list(test_dict[subnames[j]]))
                sessnum_other = len(sessnames_other)
                for sess_other_ix in range(sessnum_other):
                    # rltPath_other = f"{atlasPath}/{subnames[j]}_{sessnames_other[sess_other_ix]}_{hemi}_10k.label.gii"
                    rltPath_other = "{}/{}_{}".format(atlasPath, subnames[j], sessnames_other[sess_other_ix])+file_general
                    if not os.path.isfile(rltPath_other):
                        continue
                    rlt_other = surface.load_surf_data(rltPath_other)

                    rlt_rltOther = calDiceROI(rlt,rlt_other)
                    print("{} {} and {} {} interSubject is: {}".format(subnames[i], sessnames[sess_ix],subnames[j], sessnames_other[sess_other_ix], rlt_rltOther))
                    inter_df = inter_df.append({"sub_name":subnames[i], "sess_num":sessnames[sess_ix],"sub_other": subnames[j], "sess_other": sessnames_other[sess_other_ix], "dice":rlt_rltOther},ignore_index=True)
                    interSubject.append(rlt_rltOther)
    intra_df.to_csv("intra_df_{}_lpa.csv".format(hemi))
    inter_df.to_csv("inter_df_{}_lpa.csv".format(hemi))
    group_df.to_csv("igroup_df_{}_lpa.csv".format(hemi))

    return np.mean(intraSubject),np.std(intraSubject),np.mean(interSubject),np.std(interSubject)

def plotDiceBar(atlasPath,test_dict,hemi,output):
    m1,s1,m2,s2 = diceStats(atlasPath,test_dict,hemi)
    plt.bar(np.arange(2), [m1,m2],yerr = [s1,s2])
    plt.ylabel('dice')
    plt.xticks(np.arange(2), ('intra', 'inter'))
    plt.yticks(np.arange(0, 1, 0.2))
    plt.savefig('{}/dice_{}.jpg'.format(output, hemi))
file_path = "{}/output20230926_125936".format(os.getcwd())

cur_dir = os.getcwd()
os.chdir(file_path)
filelist = os.popen('find -name "*.label.gii"').read()
os.chdir(cur_dir)
filelist=filelist.split("\n")[:-1]

test_dict = defaultdict(set)
for filestruct in filelist:
    try:
        file_keys = filestruct.split("_")
        subname = file_keys[0].split("/")[1]
        sess_no = file_keys[1].split(".")[0]
        test_dict[subname] = test_dict[subname]|{sess_no}
    except:
        pass

plotDiceBar(file_path, test_dict, "L",".")
plotDiceBar(file_path, test_dict, "R",".")
print("ok")

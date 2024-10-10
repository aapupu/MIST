import numpy as np
import pandas as pd
import random
import os

import torch
import torch.nn as nn
import torch.nn.init as init

def make_gene_dict(gene):
    padding_dict = {'X': 0}
    gene.sort()
    gene_dict=dict(zip(gene,list(range(1,len(gene)+1))))
    padding_dict.update(gene_dict)
    return padding_dict

def make_aa_dict():
    amino_acid_dict = {
        'X': 0,
        'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5,
        'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
        'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15,
        'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20,
        "Z": 21,
    }
    return amino_acid_dict

def make_TCR_dict(species='hsa'):
    # current_dir = os.path.dirname(__file__)
    # project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    # imgt_vdj = pd.read_csv(os.path.join(project_root, 'doc', 'imgt_pip_vdj.csv')) 
    # imgt_vdj['Gene'] = imgt_vdj['0'].apply(lambda x: x.replace('DV', '/DV').replace('OR', '/OR') if ('DV' in x) or ('OR' in x) else x)
    # bv_dict = make_gene_dict(list(imgt_vdj['Gene'][imgt_vdj['Gene'].str.contains('TRBV')]))
    # bj_dict = make_gene_dict(list(imgt_vdj['Gene'][imgt_vdj['Gene'].str.contains('TRBJ')]))
    # av_dict = make_gene_dict(list(imgt_vdj['Gene'][imgt_vdj['Gene'].str.contains('TRAV')]))
    # aj_dict = make_gene_dict(list(imgt_vdj['Gene'][imgt_vdj['Gene'].str.contains('TRAJ')]))
    aa_dict = make_aa_dict()
    bv_dict_hsa = {
        'X': 0,'TRBV1': 1,'TRBV10-1': 2,'TRBV10-2': 3,'TRBV10-3': 4,'TRBV11-1': 5,'TRBV11-2': 6,'TRBV11-3': 7,'TRBV12-1': 8,'TRBV12-2': 9,
        'TRBV12-3': 10,'TRBV12-4': 11,'TRBV12-5': 12,'TRBV13': 13,'TRBV14': 14,'TRBV15': 15,'TRBV16': 16,'TRBV17': 17,'TRBV18': 18,'TRBV19': 19,
        'TRBV2': 20,'TRBV20-1': 21,'TRBV20/OR9-2': 22,'TRBV21-1': 23,'TRBV21/OR9-2': 24,'TRBV22-1': 25,'TRBV22/OR9-2': 26,'TRBV23-1': 27,'TRBV23/OR9-2': 28,
        'TRBV24-1': 29,'TRBV24/OR9-2': 30,'TRBV25-1': 31,'TRBV26': 32,'TRBV26/OR9-2': 33,'TRBV27': 34,'TRBV28': 35,'TRBV29-1': 36,'TRBV29/OR9-2': 37,'TRBV3-1': 38,
        'TRBV30': 39,'TRBV4-1': 40,'TRBV4-2': 41,'TRBV4-3': 42,'TRBV5-1': 43,'TRBV5-2': 44,'TRBV5-3': 45,'TRBV5-4': 46,'TRBV5-5': 47,'TRBV5-6': 48,'TRBV5-7': 49,
        'TRBV5-8': 50,'TRBV6-1': 51,'TRBV6-2': 52,'TRBV6-4': 53,'TRBV6-5': 54,'TRBV6-6': 55,'TRBV6-7': 56,'TRBV6-8': 57,'TRBV6-9': 58,'TRBV7-1': 59,'TRBV7-2': 60,
        'TRBV7-3': 61,'TRBV7-4': 62,'TRBV7-5': 63,'TRBV7-6': 64,'TRBV7-7': 65,'TRBV7-8': 66,'TRBV7-9': 67,'TRBV8-1': 68,'TRBV8-2': 69,'TRBV9': 70,'TRBVA': 71,
        'TRBVB': 72}
    bj_dict_hsa = {
        'X': 0,'TRBJ1-1': 1,'TRBJ1-2': 2,'TRBJ1-3': 3,'TRBJ1-4': 4,'TRBJ1-5': 5,'TRBJ1-6': 6,'TRBJ2-1': 7,'TRBJ2-2': 8,'TRBJ2-2P': 9,'TRBJ2-3': 10,
        'TRBJ2-4': 11,'TRBJ2-5': 12,'TRBJ2-6': 13,'TRBJ2-7': 14}
    av_dict_hsa = {
        'X': 0,'TRAV1-1': 1,'TRAV1-2': 2,'TRAV10': 3,'TRAV11': 4,'TRAV12-1': 5,'TRAV12-2': 6,'TRAV12-3': 7,'TRAV13-1': 8,'TRAV13-2': 9,
        'TRAV14/DV4': 10,'TRAV15': 11,'TRAV16': 12,'TRAV17': 13,'TRAV18': 14,'TRAV19': 15,'TRAV2': 16,'TRAV20': 17,'TRAV21': 18,
        'TRAV22': 19,'TRAV23/DV6': 20,'TRAV24': 21,'TRAV25': 22,'TRAV26-1': 23,'TRAV26-2': 24,'TRAV27': 25,'TRAV28': 26,
        'TRAV29/DV5': 27,'TRAV3': 28,'TRAV30': 29,'TRAV31': 30,'TRAV32': 31,'TRAV33': 32,'TRAV34': 33,'TRAV35': 34,
        'TRAV36/DV7': 35,'TRAV37': 36,'TRAV38-1': 37,'TRAV38-2/DV8': 38,'TRAV39': 39,'TRAV4': 40,'TRAV40': 41,
        'TRAV41': 42,'TRAV5': 43,'TRAV6': 44,'TRAV7': 45,'TRAV8-1': 46,'TRAV8-2': 47,'TRAV8-3': 48,
        'TRAV8-4': 49,'TRAV8-5': 50,'TRAV8-6': 51,'TRAV8-7': 52,'TRAV9-1': 53,'TRAV9-2': 54}
    aj_dict_hsa = {
        'X': 0,'TRAJ1': 1,'TRAJ10': 2,'TRAJ11': 3,'TRAJ12': 4,'TRAJ13': 5,'TRAJ14': 6,'TRAJ15': 7,'TRAJ16': 8,
        'TRAJ17': 9,'TRAJ18': 10,'TRAJ19': 11,'TRAJ2': 12,'TRAJ20': 13,'TRAJ21': 14,'TRAJ22': 15,'TRAJ23': 16,
        'TRAJ24': 17,'TRAJ25': 18,'TRAJ26': 19,'TRAJ27': 20,'TRAJ28': 21,'TRAJ29': 22,'TRAJ3': 23,'TRAJ30': 24,
        'TRAJ31': 25,'TRAJ32': 26,'TRAJ33': 27,'TRAJ34': 28,'TRAJ35': 29,'TRAJ36': 30,'TRAJ37': 31,'TRAJ38': 32,
        'TRAJ39': 33,'TRAJ4': 34,'TRAJ40': 35,'TRAJ41': 36,'TRAJ42': 37,'TRAJ43': 38,'TRAJ44': 39,'TRAJ45': 40,
        'TRAJ46': 41,'TRAJ47': 42,'TRAJ48': 43,'TRAJ49': 44,'TRAJ5': 45,'TRAJ50': 46,'TRAJ51': 47,'TRAJ52': 48,
        'TRAJ53': 49,'TRAJ54': 50,'TRAJ55': 51,'TRAJ56': 52,'TRAJ57': 53,'TRAJ58': 54,'TRAJ59': 55,'TRAJ6': 56,
        'TRAJ60': 57,'TRAJ61': 58,'TRAJ7': 59,'TRAJ8': 60,'TRAJ9': 61}
    
    bv_dict_mfa = {
        'X': 0,'TRBV1-1': 1,'TRBV1-2': 2,'TRBV1-3': 3,'TRBV10-1': 4,'TRBV10-2': 5,'TRBV10-3': 6,'TRBV11-1': 7,'TRBV11-2': 8,
        'TRBV11-3': 9,'TRBV12-1': 10,'TRBV12-2': 11,'TRBV12-3': 12,'TRBV12-4': 13,'TRBV13': 14,'TRBV14': 15,'TRBV15': 16,'TRBV16': 17,
        'TRBV18': 18,'TRBV19': 19,'TRBV2-1': 20,'TRBV2-2': 21,'TRBV2-3': 22,'TRBV20-1': 23,'TRBV21-1': 24,'TRBV22-1': 25,'TRBV23-1': 26,
        'TRBV24-1': 27,'TRBV25-1': 28,'TRBV26': 29,'TRBV27': 30,'TRBV28': 31,'TRBV29-1': 32,'TRBV3-1': 33,'TRBV3-2': 34,'TRBV3-3': 35,
        'TRBV3-4': 36,'TRBV30': 37,'TRBV4-1': 38,'TRBV4-2': 39,'TRBV4-3': 40,'TRBV5-1': 41,'TRBV5-10': 42,'TRBV5-2': 43,'TRBV5-3': 44,
        'TRBV5-4': 45,'TRBV5-5': 46,'TRBV5-6': 47,'TRBV5-7': 48,'TRBV5-8': 49,'TRBV5-9': 50,'TRBV6-1': 51,'TRBV6-2': 52,'TRBV6-2-1': 53,
        'TRBV6-3': 54,'TRBV6-4': 55,'TRBV6-5': 56,'TRBV6-5-1': 57,'TRBV6-6': 58,'TRBV6-7': 59,'TRBV6-8': 60,'TRBV7-1': 61,'TRBV7-10': 62,
        'TRBV7-2': 63,'TRBV7-3': 64,'TRBV7-4': 65,'TRBV7-5': 66,'TRBV7-6': 67,'TRBV7-7': 68,'TRBV7-7-1': 69,'TRBV7-8': 70,'TRBV7-9': 71,
        'TRBV8-2': 72,'TRBV9': 73,'TRBVA': 74,'TRBVB': 75}
    bj_dict_mfa = {
        'X': 0,'TRBJ1-1': 1,'TRBJ1-2': 2,'TRBJ1-3': 3,'TRBJ1-4': 4,'TRBJ1-5': 5,'TRBJ1-6': 6,'TRBJ2-1': 7,'TRBJ2-2': 8,'TRBJ2-2P': 9,'TRBJ2-3': 10,
        'TRBJ2-4': 11,'TRBJ2-5': 12,'TRBJ2-6': 13,'TRBJ2-7': 14}
    av_dict_mfa = {
        'X': 0,'TRAV1-1': 1,'TRAV1-2': 2,'TRAV2': 3,'TRAV3': 4,'TRAV4': 5,'TRAV5': 6,'TRAV6': 7,'TRAV8-1': 8,'TRAV8-2': 9,
        'TRAV8-3': 12,'TRAV8-6': 13,'TRAV9-1': 14,'TRAV9-2': 15,'TRAV10': 16,'TRAV12-2': 17,'TRAV12-3': 18,'TRAV13-1': 19,
        'TRAV13-2': 20,'TRAV16': 21,'TRAV18': 22,'TRAV19': 23,'TRAV20': 25,'TRAV21': 26,'TRAV22': 27,'TRAV23-1': 29,
        'TRAV24-1': 31,'TRAV24-2': 32,'TRAV25': 33,'TRAV26-1': 34,'TRAV26-2': 35,'TRAV27': 36,'TRAV29': 37,'TRAV30': 38,'TRAV32': 39,
        'TRAV33': 40,'TRAV34': 41,'TRAV35': 42,'TRAV36': 43,'TRAV37': 44,'TRAV38-1': 45,'TRAV38-2': 46,'TRAV39': 47,'TRAV40': 48,'TRAV41': 49,
        'TRAV46': 50,'TRAVA': 51,'TRAVB': 52}
    aj_dict_mfa = {
        'X': 0,'TRAJ1': 1,'TRAJ2': 2,'TRAJ3': 3,'TRAJ4': 4,'TRAJ5': 5,'TRAJ6': 6,'TRAJ7': 7,'TRAJ8': 8,'TRAJ9': 9,'TRAJ10': 10,
        'TRAJ11': 11,'TRAJ12': 12,'TRAJ13': 13,'TRAJ14': 14,'TRAJ15': 15,'TRAJ16': 16,'TRAJ17': 17,'TRAJ18': 18,'TRAJ19': 19,
        'TRAJ20': 20,'TRAJ21': 21,'TRAJ22': 22,'TRAJ23': 23,'TRAJ24': 24,'TRAJ25': 25,'TRAJ26': 26,'TRAJ27': 27,'TRAJ28': 28,'TRAJ29': 29,
        'TRAJ30': 30,'TRAJ31': 31,'TRAJ32': 32,'TRAJ33': 33,'TRAJ34': 34,'TRAJ35': 35,'TRAJ36': 36,'TRAJ37': 37,'TRAJ38': 38,'TRAJ39': 39,
        'TRAJ40': 40,'TRAJ41': 41,'TRAJ42': 42,'TRAJ43': 43,'TRAJ44': 44,'TRAJ45': 45,'TRAJ46': 46,'TRAJ47': 47,'TRAJ48': 48,'TRAJ49': 49,
        'TRAJ50': 50,'TRAJ51': 51,'TRAJ52': 52,'TRAJ53': 53,'TRAJ54': 54,'TRAJ55': 55,'TRAJ56': 56,'TRAJ57': 57,'TRAJ58': 58,'TRAJ59': 59,
        'TRAJ60': 60,'TRAJ61': 61}
    
    bv_dict_mmu = {
        'X': 0,'TRBV1-1': 1,'TRBV1-2': 2,'TRBV1-3': 3,'TRBV10-1': 4,'TRBV10-2': 5,'TRBV10-3': 6,'TRBV11-1': 7,'TRBV11-2': 8,'TRBV11-3': 9,
        'TRBV12-1': 10,'TRBV12-2': 11,'TRBV12-3': 12,'TRBV12-4': 13,'TRBV13': 14,'TRBV14': 15,'TRBV15': 16,'TRBV16': 17,'TRBV18': 18,'TRBV19': 19,
        'TRBV2-1': 20,'TRBV2-2': 21,'TRBV2-3': 22,'TRBV20-1': 23,'TRBV21-1': 24,'TRBV22-1': 25,'TRBV23-1': 26,'TRBV24-1': 27,'TRBV25-1': 28,'TRBV26': 29,
        'TRBV27': 30,'TRBV28': 31,'TRBV29-1': 32,'TRBV3-1': 33,'TRBV3-2': 34,'TRBV3-3': 35,'TRBV3-4': 36,'TRBV30': 37,'TRBV4-1': 38,'TRBV4-2': 39,
        'TRBV4-3': 40,'TRBV5-1': 41,'TRBV5-10': 42,'TRBV5-2': 43,'TRBV5-3': 44,'TRBV5-4': 45,'TRBV5-5': 46,'TRBV5-6': 47,'TRBV5-7': 48,'TRBV5-8': 49,
        'TRBV5-9': 50,'TRBV6-1': 51,'TRBV6-2': 52,'TRBV6-2-1': 53,'TRBV6-3': 54,'TRBV6-4': 55,'TRBV6-5': 56,'TRBV6-5-1': 57,'TRBV6-6': 58,'TRBV6-7': 59,
        'TRBV6-8': 60,'TRBV7-1': 61,'TRBV7-10': 62,'TRBV7-2': 63,'TRBV7-3': 64,'TRBV7-4': 65,'TRBV7-5': 66,'TRBV7-6': 67,'TRBV7-7': 68,'TRBV7-7-1': 69,
        'TRBV7-8': 70,'TRBV7-9': 71,'TRBV8-2': 72,'TRBV9': 73,'TRBVA': 74,'TRBVB': 75}
    bj_dict_mmu = {
        'X': 0,'TRBJ1-1': 1,'TRBJ1-2': 2,'TRBJ1-3': 3,'TRBJ1-4': 4,'TRBJ1-5': 5,'TRBJ1-6': 6,'TRBJ2-1': 7,'TRBJ2-2': 8,'TRBJ2-2P': 9,'TRBJ2-3': 10,
        'TRBJ2-4': 11,'TRBJ2-5': 12,'TRBJ2-6': 13,'TRBJ2-7': 14}
    av_dict_mmu = {
        'X': 0,'TRAV1-1': 1,'TRAV1-2': 2,'TRAV10': 3,'TRAV11-1': 4,'TRAV11-3': 5,'TRAV12-1': 6,'TRAV12-2': 7,'TRAV12-3': 8,'TRAV13-1': 9,'TRAV13-2': 10,
        'TRAV14-1': 11,'TRAV14-2': 12,'TRAV15': 13,'TRAV16': 14,'TRAV17': 15,'TRAV18': 16,'TRAV19': 17,'TRAV2': 18,'TRAV20': 19,
        'TRAV21': 20,'TRAV22-1': 21,'TRAV22-2': 22,'TRAV22-3': 23,'TRAV23-1': 24,'TRAV23-2': 25,'TRAV23-3': 26,'TRAV23-4': 27,'TRAV24': 28,'TRAV24-1': 29,'TRAV25': 30,
        'TRAV25-1': 31,'TRAV26-1': 32,'TRAV26-2': 33,'TRAV26-3': 34,'TRAV27': 35,'TRAV29': 36,'TRAV3': 37,'TRAV30': 38,'TRAV32': 39,'TRAV33': 40,
        'TRAV34': 41,'TRAV35': 42,'TRAV36': 43,'TRAV37': 44,'TRAV38-1': 45,'TRAV38-2': 46,'TRAV39': 47,'TRAV4': 48,'TRAV40': 49,'TRAV41': 50,
        'TRAV46': 51,'TRAV5': 52,'TRAV6': 53,'TRAV8-1': 54,'TRAV8-2': 55,'TRAV8-3': 56,'TRAV8-4': 57,'TRAV8-5': 58,'TRAV8-6': 59,'TRAV8-7': 60,
        'TRAV9-1': 61,'TRAV9-2': 62,'TRAVA': 63,'TRAVB': 64}
    aj_dict_mmu = {
        'X': 0,'TRAJ1': 1,'TRAJ10': 2,'TRAJ11': 3,'TRAJ12': 4,'TRAJ13': 5,'TRAJ14': 6,'TRAJ15': 7,'TRAJ16': 8,'TRAJ17': 9,'TRAJ18': 10,
        'TRAJ19': 11,'TRAJ2': 12,'TRAJ20': 13,'TRAJ21': 14,'TRAJ22': 15,'TRAJ23': 16,'TRAJ24': 17,'TRAJ25': 18,'TRAJ26': 19,'TRAJ27': 20,
        'TRAJ28': 21,'TRAJ29': 22,'TRAJ3': 23,'TRAJ30': 24,'TRAJ31': 25,'TRAJ32': 26,'TRAJ33': 27,'TRAJ34': 28,'TRAJ35': 29,'TRAJ36': 30,
        'TRAJ37': 31,'TRAJ38': 32,'TRAJ39': 33,'TRAJ4': 34,'TRAJ40': 35,'TRAJ41': 36,'TRAJ42': 37,'TRAJ43': 38,'TRAJ44': 39,'TRAJ45': 40,
        'TRAJ46': 41,'TRAJ47': 42,'TRAJ48': 43,'TRAJ49': 44,'TRAJ5': 45,'TRAJ50': 46,'TRAJ51': 47,'TRAJ52': 48,'TRAJ53': 49,'TRAJ54': 50,
        'TRAJ55': 51,'TRAJ56': 52,'TRAJ57': 53,'TRAJ58': 54,'TRAJ59': 55,'TRAJ6': 56,'TRAJ60': 57,'TRAJ61': 58,'TRAJ7': 59,'TRAJ8': 60,
        'TRAJ9': 61}
    
    bv_dict_mac = {
        'X': 0,'TRBV1-1': 1,'TRBV1-2': 2,'TRBV1-3': 3,'TRBV10-1': 4,'TRBV10-2': 5,'TRBV10-3': 6,'TRBV11-1': 7,'TRBV11-2': 8,'TRBV11-3': 9,'TRBV12-1': 10,
        'TRBV12-2': 11,'TRBV12-3': 12,'TRBV12-4': 13,'TRBV13': 14,'TRBV14': 15,'TRBV15': 16,'TRBV16': 17,'TRBV18': 18,'TRBV19': 19,'TRBV2-1': 20,
        'TRBV2-2': 21,'TRBV2-3': 22,'TRBV20-1': 23,'TRBV21-1': 24,'TRBV22-1': 25,'TRBV23-1': 26,'TRBV24-1': 27,'TRBV25-1': 28,'TRBV26': 29,'TRBV27': 30,
        'TRBV28': 31,'TRBV29-1': 32,'TRBV3-1': 33,'TRBV3-2': 34,'TRBV3-3': 35,'TRBV3-4': 36,'TRBV30': 37,'TRBV4-1': 38,'TRBV4-2': 39,'TRBV4-3': 40,
        'TRBV5-1': 41,'TRBV5-10': 42,'TRBV5-2': 43,'TRBV5-3': 44,'TRBV5-4': 45,'TRBV5-5': 46,'TRBV5-6': 47,'TRBV5-7': 48,'TRBV5-8': 49,'TRBV5-9': 50,
        'TRBV6-1': 51,'TRBV6-2': 52,'TRBV6-2-1': 53,'TRBV6-3': 54,'TRBV6-4': 55,'TRBV6-5': 56,'TRBV6-5-1': 57,'TRBV6-6': 58,'TRBV6-7': 59,'TRBV6-8': 60,
        'TRBV7-1': 61,'TRBV7-10': 62,'TRBV7-2': 63,'TRBV7-3': 64,'TRBV7-4': 65,'TRBV7-5': 66,'TRBV7-6': 67,'TRBV7-7': 68,'TRBV7-7-1': 69,'TRBV7-8': 70,
        'TRBV7-9': 71,'TRBV8-2': 72,'TRBV9': 73,'TRBVA': 74,'TRBVB': 75}
    bj_dict_mac = {
        'X': 0,'TRBJ1-1': 1,'TRBJ1-2': 2,'TRBJ1-3': 3,'TRBJ1-4': 4,'TRBJ1-5': 5,'TRBJ1-6': 6,'TRBJ2-1': 7,'TRBJ2-2': 8,'TRBJ2-2P': 9,'TRBJ2-3': 10,
        'TRBJ2-4': 11,'TRBJ2-5': 12,'TRBJ2-6': 13,'TRBJ2-7': 14}
    av_dict_mac = {
        'X': 0,'TRAV1-1': 1,'TRAV1-2': 2,'TRAV10': 3,'TRAV11-1': 4,'TRAV11-3': 5,'TRAV12-1': 6,'TRAV12-2': 7,'TRAV12-3': 8,'TRAV13-1': 9,'TRAV13-2': 10,
        'TRAV14-1': 11,'TRAV14-2': 12,'TRAV15': 13,'TRAV16': 14,'TRAV17': 15,'TRAV18': 16,'TRAV19': 17,'TRAV2': 18,'TRAV20': 19,'TRAV21': 20,
        'TRAV22': 21,'TRAV22-1': 22,'TRAV22-2': 23,'TRAV22-3': 24,'TRAV23-1': 25,'TRAV23-2': 26,'TRAV23-3': 27,'TRAV23-4': 28,'TRAV24': 29,'TRAV24-1': 30,
        'TRAV24-2': 31,'TRAV25': 32,'TRAV25-1': 33,'TRAV26-1': 34,'TRAV26-2': 35,'TRAV26-3': 36,'TRAV27': 37,'TRAV29': 38,'TRAV3': 39,'TRAV30': 40,
        'TRAV32': 41,'TRAV33': 42,'TRAV34': 43,'TRAV35': 44,'TRAV36': 45,'TRAV37': 46,'TRAV38-1': 47,'TRAV38-2': 48,'TRAV39': 49,'TRAV4': 50,
        'TRAV40': 51,'TRAV41': 52,'TRAV46': 53,'TRAV5': 54,'TRAV6': 55,'TRAV8-1': 56,'TRAV8-2': 57,'TRAV8-3': 58,'TRAV8-4': 59,'TRAV8-5': 60,
        'TRAV8-6': 61,'TRAV8-7': 62,'TRAV9-1': 63,'TRAV9-2': 64,'TRAVA': 65,'TRAVB': 66}
    aj_dict_mac = {
        'X': 0,'TRAJ1': 1,'TRAJ10': 2,'TRAJ11': 3,'TRAJ12': 4,'TRAJ13': 5,'TRAJ14': 6,'TRAJ15': 7,'TRAJ16': 8,'TRAJ17': 9,'TRAJ18': 10,
        'TRAJ19': 11,'TRAJ2': 12,'TRAJ20': 13,'TRAJ21': 14,'TRAJ22': 15,'TRAJ23': 16,'TRAJ24': 17,'TRAJ25': 18,'TRAJ26': 19,'TRAJ27': 20,
        'TRAJ28': 21,'TRAJ29': 22,'TRAJ3': 23,'TRAJ30': 24,'TRAJ31': 25,'TRAJ32': 26,'TRAJ33': 27,'TRAJ34': 28,'TRAJ35': 29,'TRAJ36': 30,
        'TRAJ37': 31,'TRAJ38': 32,'TRAJ39': 33,'TRAJ4': 34,'TRAJ40': 35,'TRAJ41': 36,'TRAJ42': 37,'TRAJ43': 38,'TRAJ44': 39,'TRAJ45': 40,
        'TRAJ46': 41,'TRAJ47': 42,'TRAJ48': 43,'TRAJ49': 44,'TRAJ5': 45,'TRAJ50': 46,'TRAJ51': 47,'TRAJ52': 48,'TRAJ53': 49,'TRAJ54': 50,
        'TRAJ55': 51,'TRAJ56': 52,'TRAJ57': 53,'TRAJ58': 54,'TRAJ59': 55,'TRAJ6': 56,'TRAJ60': 57,'TRAJ61': 58,'TRAJ7': 59,'TRAJ8': 60,
        'TRAJ9': 61}
    
    if species=='hsa':
        bv_dict = bv_dict_hsa
        bj_dict = bj_dict_hsa
        av_dict = av_dict_hsa
        aj_dict = aj_dict_hsa
    elif species=='mfa':
        bv_dict = bv_dict_mfa
        bj_dict = bj_dict_mfa
        av_dict = av_dict_mfa
        aj_dict = aj_dict_mfa     
    elif species=='mmu':
        bv_dict = bv_dict_mmu
        bj_dict = bj_dict_mmu
        av_dict = av_dict_mmu
        aj_dict = aj_dict_mmu
    elif species=='mac':
        bv_dict = bv_dict_mac
        bj_dict = bj_dict_mac
        av_dict = av_dict_mac
        aj_dict = aj_dict_mac
    else:
        print('Unsupported species! Supported: hsa, mfa, mmu, mac.')
        exit(1)
    return {'AA':aa_dict, 'TRBV':bv_dict, 'TRBJ':bj_dict,
            'TRAV':av_dict, 'TRAJ':aj_dict}

def gene_to_vec(genelist, gene_dict):
    return np.array([gene_dict[gene] for gene in genelist])

def aa_to_vec(aa_seq, aa_dict):
    vec = np.array([aa_dict[aa] for aa in aa_seq])
    return vec

def cdr3_to_vec(cdr3, aa_dict, max_len=30, end=False):
    cdr3 = cdr3.replace(u'\xa0', u'').upper()
    k = len(cdr3)
    if k > max_len:
        raise ValueError(f'cdr3 {cdr3} has length {len(cdr3)} > 30.')
    if end == True:
        cdr3_padding = cdr3 + "Z" + "X" * (max_len - k)
    else:
        cdr3_padding = cdr3 + "X" * (max_len - k)
    vec = aa_to_vec(cdr3_padding, aa_dict)
    return vec

def tcr_to_vec(adata,  aa_dict=None, 
               bv_dict=None, bj_dict=None, 
               av_dict=None, aj_dict=None, 
               max_len=30):
    bv = gene_to_vec(adata.obs['IR_VDJ_1_v_call'].tolist(), bv_dict)
    bj = gene_to_vec(adata.obs['IR_VDJ_1_j_call'].tolist(), bj_dict)
    cdr3b = np.array([cdr3_to_vec(cdr3, aa_dict, max_len) for cdr3 in adata.obs['IR_VDJ_1_junction_aa'].tolist()])
    
    av = gene_to_vec(adata.obs['IR_VJ_1_v_call'].tolist(), av_dict)
    aj = gene_to_vec(adata.obs['IR_VJ_1_j_call'].tolist(), aj_dict)
    cdr3a = np.array([cdr3_to_vec(cdr3, aa_dict, max_len) for cdr3 in adata.obs['IR_VJ_1_junction_aa'].tolist()])
    return bv, bj, cdr3b, av, aj, cdr3a

def convert_to_cdr3_sequence(cdr3_gen, aa_dict, temperature=1):
    scaled_logits = cdr3_gen / temperature
    probabilities = torch.softmax(scaled_logits, dim=2)
    cdr3_indices = torch.multinomial(probabilities.view(-1, cdr3_gen.size(2)),
                                     num_samples=1).view(cdr3_gen.size(0), cdr3_gen.size(1))
    cdr3_sequence = []
    for indices in cdr3_indices:
        sequence = ''.join([list(aa_dict.keys())[index] for index in indices])
        cdr3_sequence.append(sequence)
    return cdr3_sequence

def convert_to_gene(gene_gen, gene_dict, temperature=1):
    scaled_logits = gene_gen / temperature
    probabilities = torch.softmax(scaled_logits, dim=-1)
    gene_indices = torch.multinomial(probabilities, num_samples=1)
    gene = [list(gene_dict.keys())[index] for index in gene_indices]
    return gene

def convert_to_TCR(recon_bv, recon_bj, recon_cdr3b, 
                   recon_av, recon_aj, recon_cdr3a, 
                   aa_dict, bv_dict, bj_dict, 
                    av_dict, aj_dict, temperature=1):
    """
    Convert the generated TCR components from logits to actual sequences using temperature scaling.

    Args:
        recon_bv (torch.Tensor): Logits for the reconstructed TCR beta chain variable region.
        recon_bj (torch.Tensor): Logits for the reconstructed TCR beta chain joining region.
        recon_cdr3b (torch.Tensor): Logits for the reconstructed TCR beta chain CDR3 region.
        recon_av (torch.Tensor): Logits for the reconstructed TCR alpha chain variable region.
        recon_aj (torch.Tensor): Logits for the reconstructed TCR alpha chain joining region.
        recon_cdr3a (torch.Tensor): Logits for the reconstructed TCR alpha chain CDR3 region.
        aa_dict (dict): Dictionary mapping indices to amino acids.
        bv_dict (dict): Dictionary mapping indices to TCR beta chain variable region gene names.
        bj_dict (dict): Dictionary mapping indices to TCR beta chain joining region gene names.
        av_dict (dict): Dictionary mapping indices to TCR alpha chain variable region gene names.
        aj_dict (dict): Dictionary mapping indices to TCR alpha chain joining region gene names.
        temperature (float): Temperature for temperature scaling.

    Returns:
        tuple: Tuple containing the converted TCR components as sequences.
    """
    recon_bv = convert_to_gene(recon_bv.detach().cpu(), bv_dict, temperature)
    recon_bj = convert_to_gene(recon_bj.detach().cpu(), bj_dict, temperature)
    recon_cdr3b = convert_to_cdr3_sequence(recon_cdr3b.detach().cpu(),aa_dict,temperature)
    recon_av = convert_to_gene(recon_av.detach().cpu(), av_dict, temperature)
    recon_aj = convert_to_gene(recon_aj.detach().cpu(), aj_dict, temperature)
    recon_cdr3a = convert_to_cdr3_sequence(recon_cdr3a.detach().cpu(),aa_dict,temperature)
    return recon_bv,recon_bj,recon_cdr3b,recon_av,recon_aj,recon_cdr3a

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
            
            
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

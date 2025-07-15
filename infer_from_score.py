#!/usr/bin/env python
"""
Script to compute pooled for EER from saved score file for anti-spoofing task. 

Usage:
$: python PATH_TO_SCORE_FILE PATH_TO_CKPT  SET_NAME
 
 -eval_scores_path: path to the score file 
 -eval_data_name: path to the directory that has CM protocol.
 -model_path: the path to the model checkpoint, which is used to save the output EER file. 

Example:
$: python infer_from_score.py --eval_score='Scores/LA/official_model_19LA.txt' --model_path='/your/path/to/models/model_official_LA/' --eval_data_name='spoof19LA/eval'

Author: "Yikang Wang"
Email: "wwm1995@alps-lab.org"
Reference: https://github.com/TakHemlata/SSL_Anti-spoofing/tree/main/evaluate_2021_LA.py
"""


import os, time, random, argparse, pdb, numpy as np, sys, pickle as pk

def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))
    
    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]
    
    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates

    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds

def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]

def main(args):
    args.invert=False
    # os.system(f'cp {sys.argv[0]} {args.model_path}') # backup
    if args.breakdown:
        break_worker(args)
    elif args.ifseen:
        ifseen_worker(args)
    else:
        main_worker(args)
        
def break_worker(args):
    print(f'TEST on Dataset: {args.eval_data_name}')
    print(f'socre file: {args.eval_scores_path}')
    end = time.time()

    final_dict = {}
    label_dict = {}

    with open(args.eval_scores_path, 'r') as file:
        for line in file:
            key, value = line.strip().split()
            final_dict[key] = [float(value)]
    
    with open(f'data/{args.eval_data_name}/utt2eachclass', 'r') as file:
        for line in file:
            key, label = line.strip().split()
            label_dict[key] = (str(label))

    bona_cm = []
    A07_cm = []
    A08_cm = []
    A09_cm = []
    A10_cm = []
    A11_cm = []
    A12_cm = []
    A13_cm = []
    A14_cm = []
    A15_cm = []
    A16_cm = []
    A17_cm = []
    A18_cm = []
    A19_cm = []
    for key in final_dict:
        if key not in label_dict.keys():
            continue
        if label_dict[key] == '-' or label_dict[key] == 'bonafide':
            bona_cm.append(final_dict[key])
        elif label_dict[key] == 'A07':
            A07_cm.append(final_dict[key])
        elif label_dict[key] == 'A08':
            A08_cm.append(final_dict[key])
        elif label_dict[key] == 'A09':
            A09_cm.append(final_dict[key])
        elif label_dict[key] == 'A10':
            A10_cm.append(final_dict[key])
        elif label_dict[key] == 'A11': 
            A11_cm.append(final_dict[key])
        elif label_dict[key] == 'A12':
            A12_cm.append(final_dict[key])
        elif label_dict[key] == 'A13':
            A13_cm.append(final_dict[key])
        elif label_dict[key] == 'A14':
            A14_cm.append(final_dict[key])
        elif label_dict[key] == 'A15':
            A15_cm.append(final_dict[key])
        elif label_dict[key] == 'A16':
            A16_cm.append(final_dict[key])
        elif label_dict[key] == 'A17':
            A17_cm.append(final_dict[key])
        elif label_dict[key] == 'A18':
            A18_cm.append(final_dict[key])
        elif label_dict[key] == 'A19':
            A19_cm.append(final_dict[key])
    bona_cm = np.array(bona_cm).squeeze()
    A07_cm = np.array(A07_cm).squeeze()
    A08_cm = np.array(A08_cm).squeeze()
    A09_cm = np.array(A09_cm).squeeze()
    A10_cm = np.array(A10_cm).squeeze()
    A11_cm = np.array(A11_cm).squeeze()
    A12_cm = np.array(A12_cm).squeeze()
    A13_cm = np.array(A13_cm).squeeze()
    A14_cm = np.array(A14_cm).squeeze()
    A15_cm = np.array(A15_cm).squeeze()
    A16_cm = np.array(A16_cm).squeeze()
    A17_cm = np.array(A17_cm).squeeze()
    A18_cm = np.array(A18_cm).squeeze()
    A19_cm = np.array(A19_cm).squeeze()


    if not args.invert:
        A07_eer_cm = compute_eer(bona_cm, A07_cm)[0]
        A08_eer_cm = compute_eer(bona_cm, A08_cm)[0]
        A09_eer_cm = compute_eer(bona_cm, A09_cm)[0]
        A10_eer_cm = compute_eer(bona_cm, A10_cm)[0]
        A11_eer_cm = compute_eer(bona_cm, A11_cm)[0]
        A12_eer_cm = compute_eer(bona_cm, A12_cm)[0]
        A13_eer_cm = compute_eer(bona_cm, A13_cm)[0]
        A14_eer_cm = compute_eer(bona_cm, A14_cm)[0]
        A15_eer_cm = compute_eer(bona_cm, A15_cm)[0]
        A16_eer_cm = compute_eer(bona_cm, A16_cm)[0]
        A17_eer_cm = compute_eer(bona_cm, A17_cm)[0]
        A18_eer_cm = compute_eer(bona_cm, A18_cm)[0]
        A19_eer_cm = compute_eer(bona_cm, A19_cm)[0]
    else:
        A07_eer_cm = compute_eer(-bona_cm, -A07_cm)[0]
        A08_eer_cm = compute_eer(-bona_cm, -A08_cm)[0]
        A09_eer_cm = compute_eer(-bona_cm, -A09_cm)[0]
        A10_eer_cm = compute_eer(-bona_cm, -A10_cm)[0]
        A11_eer_cm = compute_eer(-bona_cm, -A11_cm)[0]
        A12_eer_cm = compute_eer(-bona_cm, -A12_cm)[0]
        A13_eer_cm = compute_eer(-bona_cm, -A13_cm)[0]
        A14_eer_cm = compute_eer(-bona_cm, -A14_cm)[0]
        A15_eer_cm = compute_eer(-bona_cm, -A15_cm)[0]
        A16_eer_cm = compute_eer(-bona_cm, -A16_cm)[0]
        A17_eer_cm = compute_eer(-bona_cm, -A17_cm)[0]
        A18_eer_cm = compute_eer(-bona_cm, -A18_cm)[0]
        A19_eer_cm = compute_eer(-bona_cm, -A19_cm)[0]

    data_time = time.time() - end
    print(f'A16_EER: {A16_eer_cm*100:.2f}\t A19_EER: {A19_eer_cm*100:.2f} |\t A07_EER: {A07_eer_cm*100:.2f}\t A08_EER: {A08_eer_cm*100:.2f}\t A09_EER: {A09_eer_cm*100:.2f}\t A17_EER: {A17_eer_cm*100:.2f} |\t A10_EER: {A10_eer_cm*100:.2f}\t A11_EER: {A11_eer_cm*100:.2f}\t A12_EER: {A12_eer_cm*100:.2f}\t A13_EER: {A13_eer_cm*100:.2f}\t A14_EER: {A14_eer_cm*100:.2f}\t A15_EER: {A15_eer_cm*100:.2f}\t   A18_EER: {A18_eer_cm*100:.2f} |\t  Time cost: {data_time}')

    # with open(os.path.join(args.model_path,'EER.txt'), 'a+') as file:
    #     file.write(f'{os.path.basename(args.eval_scores_path)}\tEER: {eer_cm*100:.4f}\t Time cost: {data_time}\n')
    with open(os.path.join(os.path.dirname(args.eval_scores_path),'breakdown_EER.txt'), 'a+') as file:
        file.write(f'{os.path.basename(args.eval_scores_path)}\t A16_EER: {A16_eer_cm*100:.2f}\t A19_EER: {A19_eer_cm*100:.2f} |\t A07_EER: {A07_eer_cm*100:.2f}\t A08_EER: {A08_eer_cm*100:.2f}\t A09_EER: {A09_eer_cm*100:.2f}\t A17_EER: {A17_eer_cm*100:.2f} |\t A10_EER: {A10_eer_cm*100:.2f}\t A11_EER: {A11_eer_cm*100:.2f}\t A12_EER: {A12_eer_cm*100:.2f}\t A13_EER: {A13_eer_cm*100:.2f}\t A14_EER: {A14_eer_cm*100:.2f}\t A15_EER: {A15_eer_cm*100:.2f}\t   A18_EER: {A18_eer_cm*100:.2f} |\t  Time cost: {data_time}\n')

def ifseen_worker(args):
    print(f'TEST on Dataset: {args.eval_data_name}')
    print(f'socre file: {args.eval_scores_path}')
    end = time.time()

    final_dict = {}
    label_dict = {}

    with open(args.eval_scores_path, 'r') as file:
        for line in file:
            key, value = line.strip().split()
            final_dict[key] = [float(value)]
    
    with open(f'data/{args.eval_data_name}/utt2eachclass', 'r') as file:
        for line in file:
            key, label = line.strip().split()
            label_dict[key] = (str(label))

    bona_cm = []
    spf_seen = []
    spf_unseen = []
    for key in final_dict:
        if key not in label_dict.keys():
            continue
        if label_dict[key] == '-' or label_dict[key] == 'bonafide':
            bona_cm.append(final_dict[key])
        elif label_dict[key] == 'A16':
            spf_seen.append(final_dict[key])
        elif label_dict[key] == 'A19':
            spf_seen.append(final_dict[key])
        elif label_dict[key] == 'A10':
            spf_unseen.append(final_dict[key])
        elif label_dict[key] == 'A11': 
            spf_unseen.append(final_dict[key])
        elif label_dict[key] == 'A12':
            spf_unseen.append(final_dict[key])
        elif label_dict[key] == 'A13':
            spf_unseen.append(final_dict[key])
        elif label_dict[key] == 'A14':
            spf_unseen.append(final_dict[key])
        elif label_dict[key] == 'A15':
            spf_unseen.append(final_dict[key])
        elif label_dict[key] == 'A18':
            spf_unseen.append(final_dict[key])
    bona_cm = np.array(bona_cm).squeeze()
    spf_seen = np.array(spf_seen).squeeze()
    spf_unseen = np.array(spf_unseen).squeeze()


    if not args.invert:
        seen_eer_cm = compute_eer(bona_cm, spf_seen)[0]
        unseen_eer_cm = compute_eer(bona_cm, spf_unseen)[0]
    else:
        seen_eer_cm = compute_eer(bona_cm, spf_seen)[0]
        unseen_eer_cm = compute_eer(bona_cm, spf_unseen)[0]

    data_time = time.time() - end
    print(f' SEEN EER: {seen_eer_cm*100:.2f}\t UNSEEN EER: {unseen_eer_cm*100:.2f}\t Time cost: {data_time}')

    # with open(os.path.join(args.model_path,'EER.txt'), 'a+') as file:
    #     file.write(f'{os.path.basename(args.eval_scores_path)}\tEER: {eer_cm*100:.4f}\t Time cost: {data_time}\n')
    with open(os.path.join(os.path.dirname(args.eval_scores_path),'ifseen_EER.txt'), 'a+') as file:
        file.write(f'{os.path.basename(args.eval_scores_path)}\t SEEN EER/UNSEEN EER: {seen_eer_cm*100:.2f}\t {unseen_eer_cm*100:.2f}\t  Time cost: {data_time}\n')
        
def main_worker(args):
    print(f'TEST on Dataset: {args.eval_data_name}')
    print(f'socre file: {args.eval_scores_path}')
    end = time.time()

    final_dict = {}
    label_dict = {}

    with open(args.eval_scores_path, 'r') as file:
        for line in file:
            key, value = line.strip().split()
            final_dict[key] = [float(value)]
    
    with open(f'data/{args.eval_data_name}/utt2label', 'r') as file:
        for line in file:
            key, label = line.strip().split()
            label_dict[key] = (str(label))

    bona_cm = []
    spoof_cm = []
    for key in final_dict:
        if key not in label_dict.keys():
            continue
        if label_dict[key] == 'bonafide':
            bona_cm.append(final_dict[key])
        else:
            spoof_cm.append(final_dict[key])
    bona_cm = np.array(bona_cm).squeeze()
    spoof_cm = np.array(spoof_cm).squeeze()

    if not args.invert:
        eer_cm = compute_eer(bona_cm, spoof_cm)[0]
        threshold = compute_eer(bona_cm, spoof_cm)[1]
    else:
        eer_cm = compute_eer(-bona_cm, -spoof_cm)[0]
        threshold = compute_eer(-bona_cm, -spoof_cm)[1]

    data_time = time.time() - end
    print(f'EER: {eer_cm*100:.4f}\t Time cost: {data_time}\t Threshold: {threshold}\n', flush=True)    


    # with open(os.path.join(args.model_path,'EER.txt'), 'a+') as file:
    #     file.write(f'{os.path.basename(args.eval_scores_path)}\tEER: {eer_cm*100:.4f}\t Time cost: {data_time}\n')
    with open(os.path.join(os.path.dirname(args.eval_scores_path),'EER.txt'), 'a+') as file:
        file.write(f'{os.path.basename(args.eval_scores_path)}\tEER: {eer_cm*100:.4f}\t Threshold: {threshold}\t Time cost: {data_time}\n')

 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conformer ctc asr pretrained')
    parser.add_argument('--eval_scores_path',required=True)
    parser.add_argument('--eval_data_name',default='spoof19LA/eval',type=str)
    parser.add_argument('--model_path',default='conformer_mfa_medium_asr',type=str)
    parser.add_argument('--breakdown',action='store_true',help='whether to compute EER for each class',default=False)
    parser.add_argument('--ifseen',action='store_true',help='whether to compute EER for seen or unseen class',default=False)
    args = parser.parse_args()
    main(args)
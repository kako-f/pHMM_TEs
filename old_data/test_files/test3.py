import parasail
from pyfaidx import Fasta
from pathlib import Path
import os
import argparse
import subprocess
from tqdm import tqdm
import pandas as pd
import hvplot.pandas  # noqa

import numpy as np


def main(main_args):
    print(main_args)
    print('File: ', main_args.f)
    print('Workdir: ', main_args.w)
    print('TE', main_args.te)
    print('aligner', main_args.al)

    fasta_file = Fasta(main_args.f)
    hmmsearch_params = 'max'

    print('Total Sequences: ', len(fasta_file.keys()))

    work_dir = main_args.w
    sim_dir = os.path.join(work_dir, main_args.te + '_data_' + main_args.al + '_')
    # Creating folders
    Path(sim_dir).mkdir(parents=True, exist_ok=True)

    score_file_name = main_args.te + '_pw_al_all_vs_all_score.csv'
    score_file = Path(os.path.join(sim_dir, score_file_name))
    if score_file.exists():
        print('File loaded')
        pid_score_df = pd.read_csv(os.path.join(sim_dir, score_file_name), index_col=0)
    else:
        print('Calculating')
        score_id_arr = []
        for id_ in tqdm(fasta_file.keys(), desc='PW - SW Calculations'):
            for id_2 in fasta_file.keys():
                result = parasail.sw_striped_sat(str(fasta_file[id_]), str(fasta_file[id_2]), 10, 1, parasail.dnafull)
                score_id_arr.append(result.score)
        score_id_arr = np.array(score_id_arr)
        score_id_matrix = score_id_arr.reshape((len(fasta_file.keys()), len(fasta_file.keys())))
        pid_score_df = pd.DataFrame(score_id_matrix, index=list(fasta_file.keys()), columns=list(fasta_file.keys()))
        # adding a row of cummulative sum
        pid_score_df.loc['column_total'] = pid_score_df.sum(numeric_only=True, axis=0)
        pid_score_df.to_csv(score_file)

    """
    sorted data
    """
    sorted_df = pid_score_df.loc['column_total'].sort_values(ascending=False).to_frame()

    """
    folder creation
    """
    sim_al_dir = os.path.join(sim_dir, 'alignment_files')
    Path(sim_al_dir).mkdir(parents=True, exist_ok=True)
    sim_fasta_dir = os.path.join(sim_dir, 'fasta_files')
    Path(sim_fasta_dir).mkdir(parents=True, exist_ok=True)

    train_folder_f = os.path.join(sim_fasta_dir, 'train_files')
    Path(train_folder_f).mkdir(parents=True, exist_ok=True)
    test_folder_f = os.path.join(sim_fasta_dir, '')
    Path(test_folder_f).mkdir(parents=True, exist_ok=True)

    sim_hmm_dir = os.path.join(sim_dir, 'hmm_hmmbuild')
    Path(sim_hmm_dir).mkdir(parents=True, exist_ok=True)
    sim_hmm_res_dir = os.path.join(sim_dir, 'res_nhmmer')
    Path(sim_hmm_res_dir).mkdir(parents=True, exist_ok=True)

    """
    Fasta creation
    """
    for n in tqdm(range(5, len(sorted_df) + 5, 5), desc='fasta'):
        top_seqs = sorted_df.head(n)
        rest = sorted_df.iloc[n:]
        # last bit - to not have a empty file
        if len(rest):
            pass
        else:
            rest = sorted_df
        if os.path.exists(os.path.join(train_folder_f, 'train_' + str(n) + '_sequences_pw_al_score.fasta')):
            pass
        else:
            with open(os.path.join(train_folder_f, 'train_' + str(n) + '_sequences_pw_al_score.fasta'), 'w') as file:
                for id_ in top_seqs.index:
                    file.write('>' + str(id_) + '\n')
                    file.write(str(fasta_file[id_]) + '\n')
        if os.path.exists(os.path.join(test_folder_f, 'test_' + str(n) + '_sequences_pw_al_score.fasta')):
            pass
        else:
            with open(os.path.join(test_folder_f, 'test_' + str(n) + '_sequences_pw_al_score.fasta'), 'w') as file:
                for id_ in rest.index:
                    file.write('>' + str(id_) + '\n')
                    file.write(str(fasta_file[id_]) + '\n')
    """
    MAFFT alignment
    """
    for path in tqdm(Path(train_folder_f).glob('*.fasta'), desc='training fasta files'):
        if os.path.exists(sim_al_dir + '/' + str(path.stem) + '.aln'):
            pass
        else:
            mafft_cmd = 'mafft --quiet --thread -1 --localpair --reorder --maxiterate 1000 ' + str(path) + ' > ' + sim_al_dir + '/' + str(
                path.stem) + '.aln'
            # print(mafft_cmd)
            subprocess.run(mafft_cmd, shell=True)
    """
    HMM Build
    """
    for path in tqdm(Path(sim_al_dir).glob('*.aln'), desc='HMM Build'):
        base_name = str(path.stem)
        hmmbuild_cmd = 'hmmbuild --cpu 6 -o ' + sim_hmm_dir + '/' + base_name + '.log ' + sim_hmm_dir + '/' + base_name + '_profile.hmm ' + sim_al_dir + '/' + base_name + '.aln'
        # print(hmmbuild_cmd)
        subprocess.run(hmmbuild_cmd, shell=True)

    """
    HMM SEARCH
    """
    for n in tqdm(range(5, len(fasta_file.keys()) + 5, 5), desc='HMM Search'):
        base_name = 'top_' + str(n) + '_sequences_pw_al_score'
        top_seqs_file = os.path.join(sim_fasta_dir, base_name + '.fasta')

        # nhmmer - total seqs
        nhmmmer_cmd = 'nhmmer --max --cpu 6 --watson -o ' + sim_hmm_res_dir + '/' + base_name + '.log --tblout ' + sim_hmm_res_dir + '/' + base_name + '.out ' + sim_hmm_dir + '/' + base_name.replace(
            'top', 'train') + '_profile.hmm ' + str(main_args.f)

        if os.path.exists(sim_hmm_res_dir + '/' + base_name + '.log'):
            pass
        else:
            subprocess.run(nhmmmer_cmd, shell=True)
    """
    HMM SEARCH
    Testing files
    """
    hmm_res_test_dir = os.path.join(sim_hmm_res_dir, 'test_files_nhmmer')
    Path(hmm_res_test_dir).mkdir(parents=True, exist_ok=True)
    for path in tqdm(Path(test_folder_f).glob('**/*.fasta'), desc='Test files'):
        name_ = str(Path(path).stem)
        n = [int(s) for s in name_.split('_')[1].split('.') if s.isdigit()][0]
        base_name = 'top_' + str(n) + '_sequences_pw_al_score'

        nhmmmer_cmd = 'nhmmer --max --cpu 6 --watson -o ' + hmm_res_test_dir + '/' + base_name + '.log --tblout ' + hmm_res_test_dir + '/' + base_name + '.out ' + sim_hmm_dir + '/' + base_name.replace(
            'top', 'train') + '_profile.hmm ' + str(path)

        if os.path.exists(hmm_res_test_dir + '/' + base_name + '.log'):
            pass
        else:
            subprocess.run(nhmmmer_cmd, shell=True)

    sim_hmm_data = {
        'base_seqs': [],
        'total_seqs': [],
        'valid_seqs_001': [],
        'valid_seqs_01': [],
        'train_seqs': [],
        'train_valid_seqs_001': [],
        'train_valid_seqs_01': []
    }
    for path in tqdm(Path(train_folder_f).glob('**/*.fasta'), desc='Fasta files'):
        # print('fasta', path)
        r_fasta_file = Fasta(path)
        sim_hmm_data['base_seqs'].append(len(r_fasta_file.keys()))
    for path in tqdm(Path(sim_hmm_res_dir).glob('*.out'), desc='Total HMM result'):
        # print('full_hmm', path)
        df = pd.read_table(path,
                           sep=' ',
                           skipinitialspace=True,
                           comment='#',
                           header=None,
                           usecols=[0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                           names=['target_name', 'hmmfrom', 'hmmto', 'alifrom', 'alito', 'envfrom', 'envto', 'sqlen',
                                  'strand', 'evalue', 'score', 'bias']
                           )
        sim_hmm_data['total_seqs'].append(len(df))
        sim_hmm_data['valid_seqs_001'].append(len(df[(df['evalue'] < 0.01)]))
        sim_hmm_data['valid_seqs_01'].append(len(df[(df['evalue'] < 0.1)]))

    for path in tqdm(Path(hmm_res_test_dir).glob('*.out'), desc='Test HMM result'):
        # print('full_hmm', path)
        df = pd.read_table(path,
                           sep=' ',
                           skipinitialspace=True,
                           comment='#',
                           header=None,
                           usecols=[0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                           names=['target_name', 'hmmfrom', 'hmmto', 'alifrom', 'alito', 'envfrom', 'envto', 'sqlen',
                                  'strand', 'evalue', 'score', 'bias']
                           )
        sim_hmm_data['train_seqs'].append(len(df))
        sim_hmm_data['train_valid_seqs_001'].append(len(df[(df['evalue'] < 0.01)]))
        sim_hmm_data['train_valid_seqs_01'].append(len(df[(df['evalue'] < 0.1)]))
    # Creating an unique DF
    df_simhmm_data = pd.DataFrame.from_dict(sim_hmm_data)
    df_simhmm_data.sort_values(by=['base_seqs'], inplace=True)
    df_simhmm_data.set_index('base_seqs', drop=True, inplace=True)
    df_simhmm_data = df_simhmm_data.assign(file_seqs=len(fasta_file.keys()))
    df_simhmm_data.to_csv(os.path.join(sim_dir, 'sim_res_max_w_test_.csv'))

    total_file_df = df_simhmm_data[['total_seqs', 'valid_seqs_001', 'valid_seqs_01', 'file_seqs']]
    test_file_df = df_simhmm_data[['train_seqs', 'train_valid_seqs_001', 'train_valid_seqs_01', 'file_seqs']]

    tot_f_plot = total_file_df.hvplot.line(rot=90,
                                           width=1000,
                                           height=500,
                                           grid=True,
                                           xlabel='N° Sequences',
                                           ylabel='HMM Hits',
                                           group_label='Legend',
                                           title='HMM hits\n' + main_args.al + ' based HMM models.\n' + main_args.te + ' data\n' + hmmsearch_params + ' params (search)\nComplete Seq file', )
    hvplot.save(tot_f_plot, os.path.join(sim_dir, 'total_file_plot_max_.html'))
    test_f_plot = test_file_df.hvplot.line(rot=90,
                                           width=1000,
                                           height=500,
                                           grid=True,
                                           xlabel='N° Sequences',
                                           ylabel='HMM Hits',
                                           group_label='Legend',
                                           title='HMM hits\n' + main_args.al + ' based HMM models.\n' + main_args.te + ' data\n' + hmmsearch_params + ' params (search)\nTest Seq files', )
    hvplot.save(test_f_plot, os.path.join(sim_dir, 'test_files_plot_max.html'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--f', '--file', required=True)
    parser.add_argument('--w', '--workdir', required=True)

    parser.add_argument('--te', '--tesf', required=True)
    parser.add_argument('--al', '--align', required=True)

    args = parser.parse_args()

    main(args)

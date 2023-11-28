import argparse
import os
import sys
from pathlib import Path
from timeit import default_timer as timer
from datetime import timedelta
from pyfaidx import Fasta
import parasail
from tqdm import tqdm
import pandas as pd
import hvplot.pandas
import numpy as np
import subprocess
import glob
import uuid

module_path = str(Path.cwd() / "modules")
if module_path not in sys.path:
    sys.path.append(module_path)
from modules import functions

__author__ = "Camilo Fuentes-Beals"
__version__ = "1.5"
__title__ = "MSACalcs"
__license__ = "GPLv3"
__author_email__ = "kmilo.f@gmail.com"


def plots(dataframe, aligner, te_name, hmm_param, sim_dir):
    """
    Plotting the pHMM hits.
    In the complete file and the testing ones.

    :param dataframe: Pandas DF with the information of the hits of each pHMM
    :param aligner: Name of the used aligner - for information purposes.
    :param te_name: Name of the TE super family
    :param hmm_param: Extra parameters of the HMM - for information purposes
    :param sim_dir: Output directory.
    :return:
    """

    total_file_df = dataframe[['base_seqs_1', 'total_seqs', 'valid_seqs_001', 'valid_seqs_01', 'file_seqs']].copy()
    test_file_df = dataframe[
        ['base_seqs_1', 'test_seqs', 'test_valid_seqs_001', 'test_valid_seqs_01', 'file_seqs']].copy()
    total_file_df.rename(columns={
        'total_seqs': 'Found seqs.',
        'valid_seqs_001': 'Seqs w/ e-value < 0.01',
        'valid_seqs_01': 'Seqs w/ e-value < 0.1',
        'file_seqs': 'Total seqs.'
    })
    test_file_df.rename(columns={
        'test_seqs': 'Found seqs.',
        'test_valid_seqs_001': 'Seqs w/ e-value < 0.01',
        'test_valid_seqs_01': 'Seqs w/ e-value < 0.1',
        'file_seqs': 'Total seqs.'
    })
    tot_f_plot = total_file_df.hvplot.line(x='base_seqs_1',
                                           rot=90,
                                           width=1000,
                                           height=500,
                                           grid=True,
                                           xlabel='N° Sequences',
                                           ylabel='HMM Hits',
                                           group_label='Legend',
                                           title='HMM hits\n' + aligner + ' based HMM models.\n' + te_name + ' data\n' + hmm_param + ' params (search)\nComplete Seq file', )
    hvplot.save(tot_f_plot, os.path.join(sim_dir, 'total_file_plot_max_.html'))

    test_f_plot = test_file_df.hvplot.line(x='base_seqs_1',
                                           rot=90,
                                           width=1000,
                                           height=500,
                                           grid=True,
                                           xlabel='N° Sequences',
                                           ylabel='HMM Hits',
                                           group_label='Legend',
                                           title='HMM hits\n' + aligner + ' based HMM models.\n' + te_name + ' data\n' + hmm_param + ' params (search)\nTest Seq files', )
    hvplot.save(test_f_plot, os.path.join(sim_dir, 'test_files_plot_max.html'))


def results(train_folder, hmm_res_test_dir, hmm_res_dir, total_seqs):
    sim_hmm_data = {
        'base_seqs_0': [],
        'base_seqs_1': [],
        'total_seqs': [],
        'valid_seqs_001': [],
        'valid_seqs_01': [],
        'test_seqs': [],
        'test_valid_seqs_001': [],
        'test_valid_seqs_01': []
    }
    print('\nAnalyzing results, creating plots. ')
    files = sorted(glob.glob(train_folder + '/*.fasta'), key=lambda x: int(os.path.basename(x).split('_')[0]))
    for path in tqdm(files, desc='Length of fasta files.'):
        name_ = str(Path(path).stem)
        n = [int(s) for s in name_.split('_') if s.isdigit()]
        sim_hmm_data['base_seqs_0'].append(n[0])
        sim_hmm_data['base_seqs_1'].append(n[1])

    files = sorted(glob.glob(hmm_res_test_dir + '/*.out'), key=lambda x: int(os.path.basename(x).split('_')[0]))
    for path in tqdm(files, desc='pHMM Test files.'):
        df = pd.read_table(path,
                           sep=' ',
                           skipinitialspace=True,
                           comment='#',
                           header=None,
                           usecols=[0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                           names=['target_name', 'hmmfrom', 'hmmto', 'alifrom', 'alito', 'envfrom', 'envto', 'sqlen',
                                  'strand', 'evalue', 'score', 'bias']
                           )
        sim_hmm_data['test_seqs'].append(len(df))
        sim_hmm_data['test_valid_seqs_01'].append(len(df[(df['evalue'] < 0.1)]))
        sim_hmm_data['test_valid_seqs_001'].append(len(df[(df['evalue'] < 0.01)]))

    files = sorted(glob.glob(hmm_res_dir + '/*.out'), key=lambda x: int(os.path.basename(x).split('_')[0]))
    for path in tqdm(files, desc='pHMM Total file result.'):
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

    # Creating an unique DF
    df_simhmm_data = pd.DataFrame.from_dict(sim_hmm_data)
    df_simhmm_data.sort_values(by=['base_seqs_0'], inplace=True)
    # df_simhmm_data.set_index('base_seqs_0', drop=True, inplace=True)
    df_simhmm_data = df_simhmm_data.assign(file_seqs=total_seqs)

    return df_simhmm_data


def hmm_search(hmm_dir, cpu_cores, hmm_res_dir, max_param, top_strand, full_fasta_file, test_folder, step,
               hmm_res_test_dir):
    """

    :param hmm_dir:
    :param cpu_cores:
    :param hmm_res_dir:
    :param max_param:
    :param top_strand:
    :param full_fasta_file:
    :param test_folder:
    :param step:
    :param hmm_res_test_dir:
    :return:
    """
    print('\nSearching with all pHMM in the complete sequence file.')
    for path in tqdm(Path(hmm_dir).glob('*.hmm'), desc='pHMM search in complete file.'):
        nhmmer_cmd = 'nhmmer --cpu ' + str(cpu_cores)
        if max_param:
            nhmmer_cmd += ' --max '
        if top_strand:
            nhmmer_cmd += ' --watson '

        base_name = path.stem
        nhmmer_cmd += ' -o ' + hmm_res_dir + '/' + base_name + '.log --tblout ' + hmm_res_dir + '/' + base_name + \
                      '.out ' + str(path) + ' ' + str(full_fasta_file)

        if os.path.exists(hmm_res_dir + '/' + base_name + '.log'):
            pass
        else:
            try:
                subprocess.run(nhmmer_cmd, shell=True, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(e)

    print('\nSearching with all pHMM in the test sequence files.')
    fasta_file = Fasta(full_fasta_file)
    seq_list = list(fasta_file.keys())
    test_files = sorted(glob.glob(test_folder + '/*.fasta'), key=lambda x: int(os.path.basename(x).split('_')[0]))
    hmm_model_files = sorted(glob.glob(hmm_dir + '/*.hmm'), key=lambda x: int(os.path.basename(x).split('_')[0]))
    for hmm_path, test_f_path in tqdm(zip(hmm_model_files, test_files[::-1]), desc='Searching in test files.'):
        nhmmer_cmd = 'nhmmer --cpu ' + str(cpu_cores)
        if max_param:
            nhmmer_cmd += ' --max '
        if top_strand:
            nhmmer_cmd += ' --watson '
        path_obj = Path(hmm_path)
        name_ = str(path_obj.stem)
        n = [int(s) for s in name_.split('_') if s.isdigit()][1]
        output_fn = str(n) + '_vs_' + str(len(seq_list) - n) + '_test_sequences'
        nhmmer_cmd += '-o ' + hmm_res_test_dir + '/' + output_fn + '.log --tblout ' + hmm_res_test_dir + '/' + output_fn + '.out ' + str(
            hmm_path) + ' ' + str(test_f_path)
        # print(nhmmer_cmd)
        if os.path.exists(hmm_res_test_dir + '/' + output_fn + '.log'):
            pass
        else:
            try:
                subprocess.run(nhmmer_cmd, shell=True, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(e)
                print('Failed to execute nhmmer - Empty files created.\n')
                with open(hmm_res_test_dir + '/' + output_fn + '.log', 'w') as fp:
                    pass
                with open(hmm_res_test_dir + '/' + output_fn + '.out', 'w') as fp:
                    pass
    print('Done.')


def hmm_build(alignment_dir, base_aln_path, hmm_dir, cpu_cores):
    base_aln_path = Path(base_aln_path)

    print('\nBuilding first pHMM.')
    first_hmmbuild_cmd = 'hmmbuild --cpu ' + str(cpu_cores) + ' -o ' + hmm_dir + '/' + str(
        base_aln_path.stem) + '_addfragments.log ' + hmm_dir + '/' + str(
        base_aln_path.stem) + '_addfragments_profile.hmm ' + str(base_aln_path.parent) + '/' + str(
        base_aln_path.stem) + '.aln'

    if os.path.exists(hmm_dir + '/' + str(base_aln_path.stem) + '.log'):
        print('File exist. ')
    else:
        print('Done. ')
        try:
            subprocess.run(first_hmmbuild_cmd, shell=True, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(e)

    print('\nBuilding the rest of the pHMM.')
    for path in tqdm(Path(alignment_dir).glob('*.aln'), desc='HMM Build'):
        base_name = str(path.stem)
        hmmbuild_cmd = 'hmmbuild --cpu ' + str(cpu_cores) + ' -o ' + hmm_dir + '/' + base_name + '.log ' + hmm_dir \
                       + '/' + base_name + '_profile.hmm ' + str(path)
        if os.path.exists(hmm_dir + '/' + base_name + '.log'):
            pass
        else:
            try:
                subprocess.run(hmmbuild_cmd, shell=True, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(e)
    print('Done.')


def alignment(train_folder, alignment_folder, debug, cpu_cores, base_align_folder, step):
    """

    :param train_folder:
    :param alignment_folder:
    :param debug:
    :param cpu_cores:
    :param base_align_folder:
    :param step:
    :return:
    """

    if cpu_cores == 0:
        cpu_cores = '-1'

    print('\nCreating base alignment. mafft - local aligment.')
    base_fasta_path = Path(os.path.join(train_folder, str(0) + '_' + str(step) + '_train_sequences_pw_al_score.fasta'))
    mafft_cmd = 'mafft --quiet --thread -1 --localpair --reorder ' + str(
        base_fasta_path) + ' > ' + base_align_folder + '/' + str(base_fasta_path.stem) + '.aln'
    if os.path.exists(base_align_folder + '/' + str(base_fasta_path.stem) + '.aln'):
        if os.stat(base_align_folder + '/' + str(base_fasta_path.stem) + '.aln').st_size > 0:
            print('File exist.')
        else:
            subprocess.run(mafft_cmd, shell=True)
            print('Done.')
    else:
        subprocess.run(mafft_cmd, shell=True)
        print('Done.')

    base_align = base_align_folder + '/' + str(base_fasta_path.stem) + '.aln'
    files = sorted(glob.glob(train_folder + '/*.fasta'), key=lambda x: int(os.path.basename(x).split('_')[0]))
    first_align = base_align
    # Skipping first file because is the base alignment.
    for path in tqdm(files[1:], desc='Alignment on training fasta files'):
        mafft_cmd = 'mafft --thread ' + str(cpu_cores)
        if not debug:
            mafft_cmd += ' --quiet'
        else:
            mafft_cmd += ''

        mafft_cmd += ' --addfragments ' + str(path) + ' --reorder ' + str(
            base_align) + ' > ' + alignment_folder + '/' + str(Path(path).stem) + '_addfragments.aln'
        base_align = alignment_folder + '/' + str(Path(path).stem) + '_addfragments.aln'
        if os.path.exists(alignment_folder + '/' + str(Path(path).stem) + '_addfragments.aln'):
            if os.stat(alignment_folder + '/' + str(Path(path).stem) + '_addfragments.aln').st_size > 0:
                pass
            else:
                subprocess.run(mafft_cmd, shell=True)
        else:
            subprocess.run(mafft_cmd, shell=True)

    return first_align


def fasta_creation(fasta_file, dataframe, train_f, test_f, step):
    """

    :param fasta_file:
    :param dataframe:
    :param train_f:
    :param test_f:
    :param step:
    :return:
    """

    seq_list = list(fasta_file.keys())
    for n in tqdm(range(0, (len(seq_list) - step) - step + 1, step), desc='fasta'):
        top_seqs = dataframe.iloc[n:n + step]
        rest = dataframe.iloc[n + step:]
        if os.path.exists(
                os.path.join(train_f, str(n) + '_' + str(n + step) + '_train_sequences_pw_al_score.fasta')):
            pass
        else:
            with open(os.path.join(train_f, str(n) + '_' + str(n + step) + '_train_sequences_pw_al_score.fasta'),
                      'w') as file:
                for id_ in top_seqs.index:
                    file.write('>' + str(id_) + '\n')
                    file.write(str(fasta_file[id_]) + '\n')
        if os.path.exists(os.path.join(test_f, str(len(rest)) + '_test_sequences_pw_al_score.fasta')):
            pass
        else:
            with open(os.path.join(test_f, str(len(rest)) + '_test_sequences_pw_al_score.fasta'), 'w') as file:
                for id_ in rest.index:
                    file.write('>' + str(id_) + '\n')
                    file.write(str(fasta_file[id_]) + '\n')


def pw_calculations(fasta_file, sim_dir, te_name):
    """
    Pairwise calculation function. Using the parasail library for speed. This library has a python-binding that
    connect the main C code and allows use of the functions in Python.

    :param fasta_file: Fasta object.
    :param sim_dir:
    :param te_name:
    :return:
    """
    score_file_name = te_name + '_pw_al_all_vs_all_score.csv'
    score_file = Path(os.path.join(sim_dir, score_file_name))

    if score_file.exists():
        print('\nPairwise score file loaded')
        pid_score_df = pd.read_csv(os.path.join(sim_dir, score_file_name), index_col=0)
    else:
        print('\nCalculating')
        score_id_arr = []
        seq_list = list(fasta_file.keys())
        for id_ in tqdm(seq_list, desc='Pairwise alignment calculations'):
            for id_2 in seq_list:
                result = parasail.sw_striped_sat(str(fasta_file[id_]), str(fasta_file[id_2]), 10, 1, parasail.dnafull)
                score_id_arr.append(result.score)
        score_id_arr = np.array(score_id_arr)
        score_id_matrix = score_id_arr.reshape((len(seq_list), len(seq_list)))
        pid_score_df = pd.DataFrame(score_id_matrix, index=list(seq_list), columns=list(seq_list))

        """
        Adding row of cumulative sum and saving.
        """
        pid_score_df.loc['column_total'] = pid_score_df.sum(numeric_only=True, axis=0)
        pid_score_df.to_csv(score_file)

    return pid_score_df


def seq_preprocessing(fasta_file, base_dir, file_name):
    """
    Fasta file preprocessing.
    Given a fasta file with 3 or more sequences, check each one of them for sequences that do not contain all the
    nucleotides. This sequences will be removed from the complete set.

    :param fasta_file: Fasta object.
    :param base_dir:
    :param file_name:
    :return:
    """
    f_fasta_list = list(fasta_file.keys())
    skipped_seqs = []

    for id_ in tqdm(fasta_file.keys(), desc='Fasta file preprocessing'):
        seq_freq = functions.get_base_frequencies(str(fasta_file[id_]))
        if any(x == 0 for x in seq_freq.values()):
            skipped_seqs.append(id_)
        else:
            pass
    clean_fasta_file = None
    if len(skipped_seqs) > 0:
        for s_id in tqdm(skipped_seqs, desc='Removing non DNA seqs'):
            f_fasta_list.remove(s_id)
        clean_fasta_file = os.path.join(base_dir, file_name + '_clean.fasta')
        with open(clean_fasta_file, 'w+') as file:
            for id_ in f_fasta_list:
                file.write('>' + str(id_) + '\n')
                file.write(str(fasta_file[id_]) + '\n')

        print('Cleaned Fasta file saved to: ', clean_fasta_file)
    else:
        print('No bad sequence found.')

    return f_fasta_list, skipped_seqs, clean_fasta_file


def main(main_args):
    """

    :param main_args:
    :return:
    """
    # TODO
    # Future work might include aligner selection.
    aligner = 'mafft'

    # normalizing a path. i.e. removing extra '/'
    base_dir = os.path.normpath(main_args.b)
    work_dir = os.path.normpath(main_args.o)

    if not os.path.exists(work_dir):
        print('\n Creating work file.')
        os.makedirs(work_dir)

    # Main arguments
    seq_step = main_args.s
    seq_limit = main_args.l
    n_cores = main_args.n

    nhmmer_max = main_args.m
    nhmmer_top = main_args.t
    # mafft_mode = main_args.ma

    print(main_args)
    print('--- Base Options ---')
    print('* Base dir: ', base_dir)
    print('* Output dir: ', work_dir)
    print('* Total sequence limit:', seq_limit)
    print('* Sequence steps:', seq_step)
    print('* Aligner', aligner)
    print('* CPU Cores to use:', n_cores)
    print('* Verbose/Debug: ', main_args.d)
    print('--- nhmmer Options ---')
    print('nhmmer max param:', nhmmer_max)
    print('nhmmer top strand param:', nhmmer_top)

    print('----------------------\n')

    f_id = str(uuid.uuid4())
    with open(os.path.join(work_dir, 'msa_execution_' + f_id + '.log'), 'w+') as file:
        file.write('Command: ' + f"{' '.join(sys.argv)}")
        file.write('\n--- Base Options ---')
        file.write('\n* Base dir: ' + str(base_dir))
        file.write('\n* Output dir: ' + str(work_dir))
        file.write('\n* Total sequence limit:' + str(seq_limit))
        file.write('\n* Sequence steps:' + str(seq_step))
        file.write('\n* Aligner: ' + str(aligner))
        file.write('\n* CPU Cores to use:' + str(n_cores))
        file.write('\n* Verbose/Debug: ' + str(main_args.d))
        file.write('\n--- nhmmer Options ---')
        file.write('\nnhmmer max param:' + str(nhmmer_max))
        file.write('\nnhmmer top strand param:' + str(nhmmer_top) + '\n')

    start = timer()
    for path in Path(base_dir).glob('**/*.fasta'):
        print('=' * 20)
        print('Main loop')
        base_path = path
        base_file_name = base_path.stem
        te_base_dir = base_path.parent

        # TODO
        # change this variable to ask for a name or suggest a default format
        te_name = str(te_base_dir).split('/')[-1]

        if nhmmer_max:
            hmmsearch_params = 'max'
        else:
            hmmsearch_params = ''

        sim_dir = os.path.join(work_dir, te_name + '_data_' + aligner + '_')
        # Creating base folder
        Path(sim_dir).mkdir(parents=True, exist_ok=True)
        # checkpoint path
        chk_file = os.path.join(sim_dir, te_name + '.chk')

        if not os.path.exists(chk_file):
            te_fasta_file = Fasta(base_path)
            if len(te_fasta_file.keys()) < int(seq_limit):
                print('Analyzing:', te_name)

                """
                Seq preprocessing
                """
                seq_list, skipped_seqs, clean_fasta_path = seq_preprocessing(fasta_file=te_fasta_file,
                                                                             base_dir=te_base_dir,
                                                                             file_name=base_file_name)
                print('Skipped sequences', len(skipped_seqs))
                print(clean_fasta_path)
                if clean_fasta_path is not None:
                    base_path = Path(clean_fasta_path)
                    te_fasta_file = Fasta(base_path)
                else:
                    pass

                """
                Pairwise Alignment calculations
                """
                pid_score_df = pw_calculations(fasta_file=te_fasta_file, sim_dir=sim_dir, te_name=te_name)
                print('Sorting the data.')
                sorted_df = pid_score_df.loc['column_total'].sort_values(ascending=False).to_frame()

                """
                 Folder structure
                """
                print('\nCreating folder structure.')
                sim_al_dir = os.path.join(sim_dir, 'alignment_files')
                Path(sim_al_dir).mkdir(parents=True, exist_ok=True)
                sim_base_dir = os.path.join(sim_al_dir, 'base_align')
                Path(sim_base_dir).mkdir(parents=True, exist_ok=True)
                sim_fasta_dir = os.path.join(sim_dir, 'fasta_files')
                Path(sim_fasta_dir).mkdir(parents=True, exist_ok=True)

                train_folder_f = os.path.join(sim_fasta_dir, 'train_files')
                Path(train_folder_f).mkdir(parents=True, exist_ok=True)
                test_folder_f = os.path.join(sim_fasta_dir, 'test_files')
                Path(test_folder_f).mkdir(parents=True, exist_ok=True)

                sim_hmm_dir = os.path.join(sim_dir, 'hmm_models')
                Path(sim_hmm_dir).mkdir(parents=True, exist_ok=True)
                sim_hmm_res_dir = os.path.join(sim_dir, 'hmm_results')
                Path(sim_hmm_res_dir).mkdir(parents=True, exist_ok=True)

                hmm_res_test_dir = os.path.join(sim_hmm_res_dir, 'hmm_res_test_files')
                Path(hmm_res_test_dir).mkdir(parents=True, exist_ok=True)

                """
                Fasta Creation
                """
                print('\nFasta files creation.')
                fasta_creation(fasta_file=te_fasta_file, dataframe=sorted_df, train_f=train_folder_f,
                               test_f=test_folder_f, step=seq_step)

                """
                Alignments
                """
                print('\nCreation of MSA.')
                base_align_file = alignment(train_folder=train_folder_f, alignment_folder=sim_al_dir, debug=main_args.d,
                                            cpu_cores=n_cores, base_align_folder=sim_base_dir, step=seq_step)
                """
                HMM Build
                """
                hmm_build(alignment_dir=sim_al_dir, base_aln_path=base_align_file, hmm_dir=sim_hmm_dir,
                          cpu_cores=n_cores)
                """
                HMM search - complete file
                """
                hmm_search(hmm_dir=sim_hmm_dir, cpu_cores=n_cores, max_param=nhmmer_max, top_strand=nhmmer_top,
                           full_fasta_file=base_path, hmm_res_dir=sim_hmm_res_dir, test_folder=test_folder_f,
                           step=seq_step, hmm_res_test_dir=hmm_res_test_dir)
                """
                Results and plots
                """
                sim_data_df = results(train_folder=train_folder_f, hmm_res_test_dir=hmm_res_test_dir,
                                      hmm_res_dir=sim_hmm_res_dir, total_seqs=len(te_fasta_file.keys()))
                sim_data_df.to_csv(os.path.join(sim_dir, 'sim_res_max_w_test_.csv'))
                plots(dataframe=sim_data_df, sim_dir=sim_dir, te_name=te_name, hmm_param=hmmsearch_params,
                      aligner=aligner)

                print('Finished analyzing ', te_name)
                print('Creating checkpoint file in ' + str(sim_dir))

                end = timer()

                with open(os.path.join(sim_dir, te_name + '.chk'), 'w+') as file:
                    file.write('Job done for TE super family ' + str(te_name) + '\n')
                    file.write('Elapsed time: ' + str(timedelta(seconds=end - start)) + '\n')
                    file.write('Skipped sequences: \n')
                    for seq in skipped_seqs:
                        file.write(seq + '\n')
                    file.write('Sequence step: ' + str(seq_step) + '\n')
                    file.write('Original fasta file: ' + str(base_path) + '\n')

                with open(os.path.join(work_dir, 'msa_execution_' + f_id + '.log'), 'a+') as file:
                    file.write('\nTE Superfamily: ' + te_name + ' done.')
            else:
                print('TE superfamily ' + te_name + ' has ' + str(
                    len(te_fasta_file.keys())) + ' sequences. More than the ' + str(
                    seq_limit) + ' imposed limit. \nPassing')
        else:
            print('Checkpoint detected, for ' + str(te_name) + ' skipping.')


if __name__ == '__main__':
    desc = 'Pipeline for MSA calculations. Based on the proposed methodology of the thesis developed to achieve the ' \
           'PhD grade. DOI: TBP .'
    parser = argparse.ArgumentParser(description=desc, epilog='By Camilo Fuentes-Beals. @cakofuentes')
    # Base arguments
    parser.add_argument('--b', '--basedir', required=True, help='Input folder.')
    parser.add_argument('--o', '--output', required=True, help='Output folder.')
    parser.add_argument('--d', '--debug', required=False, action=argparse.BooleanOptionalAction,
                        help='Whether to show or not program logs in execution.')
    parser.add_argument('--l', '--limit', type=int, default=1000, required=False,
                        help='Upper limit of number of sequences. Default 1000.')
    parser.add_argument('--s', '--step', type=int, required=False, default=5,
                        help='Default quantity for group of sequences. Default 5.')
    # cpu arguments.
    parser.add_argument('--n', '--ncore', type=int, required=False, default=0,
                        help='Default number of CPU cores to use across the pipeline. For HMMER and MAFFT. '
                             '0 dictates to use all available cpu cores')
    # nhmmer options
    parser.add_argument('--m', '--max', required=False, default=False, action='store_true',
                        help='nhmmer max param. Disabled by default.')
    parser.add_argument('--t', '--ts', required=False, default=False, action='store_true',
                        help='Top strand param for nhmmer. Disabled by default')
    # mafft options
    parser.add_argument('--ma', '--mafft_auto', required=False, default=False, action='store_true',
                        help='Automatic selection for MAFFT alignment mode. Disabled by default (local alignment).')
    args = parser.parse_args()

    main(args)

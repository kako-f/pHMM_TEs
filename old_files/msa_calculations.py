import os
import sys
from tqdm import tqdm
from pathlib import Path
import argparse
import parasail
import numpy as np
import pandas as pd
import hvplot.pandas  # noqa
from pyfaidx import Fasta
import subprocess
from timeit import default_timer as timer
from datetime import timedelta

module_path = str(Path.cwd() / "modules")
if module_path not in sys.path:
    sys.path.append(module_path)
from modules import functions

__author__ = "Camilo Fuentes-Beals"
__version__ = "1.0"
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

    total_file_df = dataframe[['total_seqs', 'valid_seqs_001', 'valid_seqs_01', 'file_seqs']].copy()
    test_file_df = dataframe[['test_seqs', 'test_valid_seqs_001', 'test_valid_seqs_01', 'file_seqs']].copy()

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
    tot_f_plot = total_file_df.hvplot.line(rot=90,
                                           width=1000,
                                           height=500,
                                           grid=True,
                                           xlabel='N° Sequences',
                                           ylabel='HMM Hits',
                                           group_label='Legend',
                                           title='HMM hits\n' + aligner + ' based HMM models.\n' + te_name + ' data\n' + hmm_param + ' params (search)\nComplete Seq file', )
    hvplot.save(tot_f_plot, os.path.join(sim_dir, 'total_file_plot_max_.html'))
    test_f_plot = test_file_df.hvplot.line(rot=90,
                                           width=1000,
                                           height=500,
                                           grid=True,
                                           xlabel='N° Sequences',
                                           ylabel='HMM Hits',
                                           group_label='Legend',
                                           title='HMM hits\n' + aligner + ' based HMM models.\n' + te_name + ' data\n' + hmm_param + ' params (search)\nTest Seq files', )
    hvplot.save(test_f_plot, os.path.join(sim_dir, 'test_files_plot_max.html'))


def results(clean_seq_list, train_folder, hmm_res_dir, hmm_test_res_dir):
    """

    :param clean_seq_list:
    :param train_folder:
    :param hmm_res_dir:
    :param hmm_test_res_dir:
    :return:
    """
    sim_hmm_data = {
        'base_seqs': [],
        'total_seqs': [],
        'valid_seqs_001': [],
        'valid_seqs_01': [],
        'test_seqs': [],
        'test_valid_seqs_001': [],
        'test_valid_seqs_01': []
    }
    for path in tqdm(Path(train_folder).glob('*.fasta'), desc='Analyzing Fasta Files'):
        r_fasta_file = Fasta(path)
        sim_hmm_data['base_seqs'].append(len(r_fasta_file.keys()))
    for path in tqdm(Path(hmm_res_dir).glob('*.out'), desc='Analyzing Total file HMM result'):
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

    for path in tqdm(Path(hmm_test_res_dir).glob('*.out'), desc='Analyzing Test file HMM result'):
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
        sim_hmm_data['test_valid_seqs_001'].append(len(df[(df['evalue'] < 0.01)]))
        sim_hmm_data['test_valid_seqs_01'].append(len(df[(df['evalue'] < 0.1)]))

    """
    Unique DF
    """
    df_simhmm_data = pd.DataFrame.from_dict(sim_hmm_data)
    df_simhmm_data.sort_values(by=['base_seqs'], inplace=True)
    df_simhmm_data.set_index('base_seqs', drop=True, inplace=True)
    df_simhmm_data = df_simhmm_data.assign(file_seqs=len(clean_seq_list))

    return df_simhmm_data


def hmm_search_test_files(test_folder, hmm_dir, hmm_test_res_dir, max_param, top_strand, cpu_cores, redo,
                          clean_seq_list):
    """

    :param test_folder:
    :param hmm_dir:
    :param hmm_test_res_dir:
    :param max_param:
    :param top_strand:
    :param cpu_cores:
    :param redo:
    :param clean_seq_list:
    :return:
    """

    if cpu_cores == 0:
        cpu_cores = os.cpu_count()

    if redo:
        print('Deleting HMM related files in assigned folder.')
        res_log = [f.unlink() for f in Path(hmm_test_res_dir).glob("*.log") if f.is_file()]
        print(str(len(res_log)) + ' files deleted. ')
        res_out = [f.unlink() for f in Path(hmm_test_res_dir).glob("*.out") if f.is_file()]
        print(str(len(res_out)) + ' files deleted. ')

    for path in tqdm(Path(test_folder).glob('**/*.fasta'), desc='HMM search on test files'):
        name_ = str(Path(path).stem)
        n = [int(s) for s in name_.split('_')[1].split('.') if s.isdigit()][0]
        base_name = 'train_' + str(len(clean_seq_list) - n) + '_sequences_pw_al_score'

        nhmmer_cmd = 'nhmmer --cpu ' + str(cpu_cores)
        if max_param:
            nhmmer_cmd += ' --max '
        if top_strand:
            nhmmer_cmd += ' --watson '

        nhmmer_cmd += '-o ' + hmm_test_res_dir + '/' + base_name + '.log --tblout ' + hmm_test_res_dir \
                      + '/' + base_name + '.out ' + hmm_dir + '/' + base_name + '_profile.hmm ' + str(path)

        if os.path.exists(hmm_test_res_dir + '/' + base_name + '.log'):
            pass
        else:
            subprocess.run(nhmmer_cmd, shell=True)


def hmm_search_complete(full_fasta_path, hmm_dir, hmm_res_dir, max_param, top_strand, cpu_cores, redo, train_folder):
    """

    :param full_fasta_path:
    :param hmm_dir:
    :param hmm_res_dir:
    :param max_param:
    :param top_strand:
    :param cpu_cores:
    :param redo:
    :param train_folder:
    :return:
    """

    if cpu_cores == 0:
        cpu_cores = os.cpu_count()

    if redo:
        print('Deleting HMM related files in assigned folder.')
        res_log = [f.unlink() for f in Path(hmm_res_dir).glob("*.log") if f.is_file()]
        print(str(len(res_log)) + ' files deleted. ')
        res_out = [f.unlink() for f in Path(hmm_res_dir).glob("*.out") if f.is_file()]
        print(str(len(res_out)) + ' files deleted. ')

    for path in tqdm(Path(train_folder).glob('*.fasta'), desc='training fasta files'):
        base_name = path.stem

        nhmmer_cmd = 'nhmmer --cpu ' + str(cpu_cores)
        if max_param:
            nhmmer_cmd += ' --max '
        if top_strand:
            nhmmer_cmd += ' --watson '

        nhmmer_cmd += '-o ' + hmm_res_dir + '/' + base_name + '.log --tblout ' + hmm_res_dir + '/' + base_name \
                      + '.out ' + hmm_dir + '/' + base_name + '_profile.hmm ' + str(full_fasta_path)

        if os.path.exists(hmm_res_dir + '/' + base_name + '.log'):
            pass
        else:
            subprocess.run(nhmmer_cmd, shell=True)


def hmm_build(alignment_dir, hmm_dir, cpu_cores):
    """

    :param alignment_dir:
    :param hmm_dir:
    :param cpu_cores:
    :return:
    """
    for path in tqdm(Path(alignment_dir).glob('*.aln'), desc='Building HMMs'):
        base_name = str(path.stem)

        hmmbuild_cmd = 'hmmbuild --cpu ' + str(cpu_cores) + ' -o ' + hmm_dir + '/' + base_name + '.log ' + hmm_dir \
                       + '/' + base_name + '_profile.hmm ' + alignment_dir + '/' + base_name + '.aln'
        if os.path.exists(hmm_dir + '/' + base_name + '.log'):
            pass
        else:
            subprocess.run(hmmbuild_cmd, shell=True)


def alignment(train_folder, alignment_folder, debug, cpu_cores, redo, mode):
    """

    :param train_folder:
    :param alignment_folder:
    :param debug:
    :param cpu_cores:
    :param redo:
    :param mode:
    :return:
    """
    if redo:
        print('Deleting alignment files in assigned folder.')
        res = [f.unlink() for f in Path(alignment_folder).glob("*.aln") if f.is_file()]
        print(str(len(res)) + ' files deleted. ')
    else:
        for path in tqdm(Path(train_folder).glob('*.fasta'), desc='Alignment on training fasta files'):

            if cpu_cores == 0:
                cpu_cores = '-1'

            mafft_cmd = 'mafft --thread ' + str(cpu_cores)
            if not debug:
                mafft_cmd += ' --quiet'
            else:
                mafft_cmd += ''
            if mode:
                align_mode = '--auto'
            else:
                align_mode = '--localpair'

            mafft_cmd += ' ' + align_mode + ' --reorder ' + str(
                path) + ' > ' + alignment_folder + '/' + str(path.stem) + '.aln'

            if os.path.exists(alignment_folder + '/' + str(path.stem) + '.aln'):
                if os.stat(alignment_folder + '/' + str(path.stem) + '.aln').st_size > 0:
                    pass
                else:
                    subprocess.run(mafft_cmd, shell=True)
            else:
                subprocess.run(mafft_cmd, shell=True)


def fasta_creation(fasta_file, dataframe, train_f, test_f, step, clean_seq_list):
    """

    :param fasta_file:
    :param dataframe:
    :param train_f:
    :param test_f:
    :param step:
    :param clean_seq_list:
    :return:
    """

    for n in tqdm(range(step, len(clean_seq_list), step), desc='Fasta creation'):
        top_seqs = dataframe.head(n)
        rest = dataframe.iloc[n:]
        if os.path.exists(os.path.join(train_f, 'train_' + str(len(top_seqs)) + '_sequences_pw_al_score.fasta')):
            pass
        else:
            with open(os.path.join(train_f, 'train_' + str(len(top_seqs)) + '_sequences_pw_al_score.fasta'),
                      'w') as file:
                for id_ in top_seqs.index:
                    file.write('>' + str(id_) + '\n')
                    file.write(str(fasta_file[id_]) + '\n')
        if os.path.exists(os.path.join(test_f, 'test_' + str(len(rest)) + '_sequences_pw_al_score.fasta')):
            pass
        else:
            with open(os.path.join(test_f, 'test_' + str(len(rest)) + '_sequences_pw_al_score.fasta'), 'w') as file:
                for id_ in rest.index:
                    file.write('>' + str(id_) + '\n')
                    file.write(str(fasta_file[id_]) + '\n')


def pw_calculations(fasta_file, sim_dir, te_name, clean_seq_list):
    """
    Pairwise calculation function. Using the parasail library for speed. This library has a python-binding that
    connect the main C code and allows use of the functions in Python.

    :param fasta_file:
    :param sim_dir:
    :param te_name:
    :param clean_seq_list:
    :return:
    """
    score_file_name = te_name + '_pw_al_all_vs_all_score.csv'
    score_file = Path(os.path.join(sim_dir, score_file_name))

    if score_file.exists():
        print('Pairwise score file loaded')
        pid_score_df = pd.read_csv(os.path.join(sim_dir, score_file_name), index_col=0)
    else:
        print('Calculating')
        score_id_arr = []
        for id_ in tqdm(clean_seq_list, desc='Pairwise alignment calculations'):
            for id_2 in clean_seq_list:
                result = parasail.sw_striped_sat(str(fasta_file[id_]), str(fasta_file[id_2]), 10, 1, parasail.dnafull)
                score_id_arr.append(result.score)
        score_id_arr = np.array(score_id_arr)
        score_id_matrix = score_id_arr.reshape((len(clean_seq_list), len(clean_seq_list)))
        pid_score_df = pd.DataFrame(score_id_matrix, index=list(clean_seq_list), columns=list(clean_seq_list))

        """
        Adding row of cumulative sum and saving.
        """
        pid_score_df.loc['column_total'] = pid_score_df.sum(numeric_only=True, axis=0)
        pid_score_df.to_csv(score_file)

    return pid_score_df


def seq_preprocessing(fasta_file):
    """
    Fasta file preprocessing.
    Given a fasta file with 3 or more sequences, check each one of them for sequences that do not contain all the
    nucleotides. This sequences will be removed from the complete set.
    :param fasta_file:
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

    for s_id in tqdm(skipped_seqs, desc='Removing non DNA seqs'):
        f_fasta_list.remove(s_id)

    return f_fasta_list, skipped_seqs


def cd_hit_preprocessing(fasta_file_path, base_name, seq_identity, mode, mem, output_folder, debug, cpu_cores):
    """

    :param fasta_file_path:
    :param base_name:
    :param seq_identity:
    :param mode:
    :param mem:
    :param output_folder:
    :param debug:
    :param cpu_cores:
    :return:
    """
    cdhit_fasta = os.path.join(output_folder, base_name + '_cdhit.fasta')

    cdhit_cmd = 'cd-hit-est -i ' + str(
        fasta_file_path) + ' -o ' + cdhit_fasta + ' -c ' + str(
        seq_identity) + ' -g ' + str(mode) + ' -d 0 -sc 1 -T ' + str(cpu_cores) + ' -M ' + str(mem)
    if not debug:

        cdhit_cmd += ' > ' + str(os.path.join(output_folder, base_name + '_cdhit.out'))
    else:
        pass

    if os.path.exists(cdhit_fasta):
        pass
    else:
        subprocess.run(cdhit_cmd, shell=True)

    return cdhit_fasta


def main(main_args):
    """
    Main function.
    Reading arguments and passing them to the corresponding functions.

    Main loop works in the following manner.
    Given the base dir, it expects a FASTA file to read the sequences from.
    This should be a multiple sequence file (+3).
    It's designed to process TE sequences. There are hardcode variables with the TE name on it.
    But as such, the main base is a DNA sequence of any sort.
    Probably this would change in future updates to be more generic.

    before starting any process, a series of checks are done to the sequences to ensure a correct analysis.
    if the option for using cd-hit is activated, this routine will execute.
    if not, a routine for cleaning the set of sequences is executed.

    After the main process is done, a checkpoint is created in the corresponding output directory. Whether this file
    is present or not, determines if the process is finished.
    :param main_args:
    :return:
    """
    # TODO
    # Future work might include aligner selection.
    aligner = 'mafft'

    # normalizing a path. i.e. removing extra '/'

    base_dir = os.path.normpath(main_args.b)
    work_dir = main_args.o

    seq_step = main_args.s
    seq_limit = main_args.l
    n_cores = main_args.n
    redo_align = main_args.ra
    redo_search = main_args.rs
    nhmmer_max = main_args.m
    nhmmer_top = main_args.t
    mafft_mode = main_args.ma

    print(main_args)
    print('--- Base Options ---')
    print('base dir: ', base_dir)
    print('output dir: ', work_dir)
    print('aligner', aligner)
    print('limit:', seq_limit)
    print('steps:', seq_step)
    print('CPU Cores to use:', n_cores)
    print('Redo - Alignment: ', redo_align)
    print('Redo - nhmmer: ', redo_search)
    print('debug', main_args.d)
    print('--- nhmmer Options ---')
    print('nhmmer max param:', nhmmer_max)
    print('nhmmer top strand param:', nhmmer_top)
    if main_args.c:
        extra = 'cdhit'
        print('--- cd-hit-est Options ---')
        print('cd-hit-est sequence indentity:', main_args.cc)
        print('cd-hit-est clustering mode max param:', main_args.cg)
        print('cd-hit-est max memory:', main_args.cm)
    else:
        extra = ''
    print('---' * 2)
    start = timer()

    for path in Path(base_dir).glob('**/*.fasta'):
        print('Main loop')
        base_path = path
        base_file_name = base_path.stem
        te_base_dir = base_path.parent

        # TODO
        # change this variable to be asked explicit for a name or suggest a default format
        te_name = str(te_base_dir).split('/')[-1]
        # -

        hmmsearch_params = 'max'
        te_fasta_file = Fasta(base_path)

        sim_dir = os.path.join(work_dir, te_name + '_data_' + aligner + '_' + extra)

        # Creating base folder
        Path(sim_dir).mkdir(parents=True, exist_ok=True)
        # checkpoint path
        chk_file = os.path.join(sim_dir, te_name + '.chk')
        if redo_align or redo_search:
            f = Path(chk_file)
            if f.is_file():
                print('Redo option detected. - Removing checkpoint. ')
                f.unlink(missing_ok=True)
            else:
                pass
        if os.path.exists(chk_file):
            print('Checkpoint detected, for ' + str(te_name) + ' skipping.')
        else:
            if len(te_fasta_file.keys()) > int(seq_limit):
                print('TE superfamily ' + te_name + ' has ' + str(
                    len(te_fasta_file.keys())) + ' sequences. More than the ' + str(
                    seq_limit) + ' imposed limit. \nPassing')
            else:
                print('Analyzing: ', te_name)

                """
                Seq preprocessing
                """
                if main_args.c:
                    print('cd-hit-est:', main_args.c)
                    print('cd-hit-est seq. identity:', main_args.cc)
                    print('cd-hit-est algorithm:', main_args.cg)
                    print('cd-hit-est memory:', main_args.cm)
                    cd_hit_folder = str(te_base_dir).replace(base_dir.split('/')[-1],
                                                             base_dir.split('/')[-1] + '_cdhit')
                    print('cd-hit-est output folder: ', cd_hit_folder)
                    Path(cd_hit_folder).mkdir(parents=True, exist_ok=True)

                    cd_hit_path = cd_hit_preprocessing(fasta_file_path=base_path, base_name=base_file_name,
                                                       seq_identity=main_args.cc, mode=main_args.cg, mem=main_args.cm,
                                                       output_folder=cd_hit_folder, debug=main_args.d,
                                                       cpu_cores=n_cores)
                    te_fasta_file = Fasta(cd_hit_path)

                seq_list, skipped_seqs = seq_preprocessing(fasta_file=te_fasta_file)
                print('Skipped sequences', len(skipped_seqs))

                """
                Pairwise Alignment calculations
                """
                pid_score_df = pw_calculations(fasta_file=te_fasta_file, sim_dir=sim_dir, te_name=te_name,
                                               clean_seq_list=seq_list)
                print('Sorting the data.')
                sorted_df = pid_score_df.loc['column_total'].sort_values(ascending=False).to_frame()
                """
                Folder structure
                """
                sim_al_dir = os.path.join(sim_dir, 'alignment_files')
                Path(sim_al_dir).mkdir(parents=True, exist_ok=True)
                sim_fasta_dir = os.path.join(sim_dir, 'fasta_files')
                Path(sim_fasta_dir).mkdir(parents=True, exist_ok=True)

                train_folder_f = os.path.join(sim_fasta_dir, 'train_files')
                Path(train_folder_f).mkdir(parents=True, exist_ok=True)
                test_folder_f = os.path.join(sim_fasta_dir, '../test_files')
                Path(test_folder_f).mkdir(parents=True, exist_ok=True)

                sim_hmm_dir = os.path.join(sim_dir, 'hmm_models')
                Path(sim_hmm_dir).mkdir(parents=True, exist_ok=True)
                sim_hmm_res_dir = os.path.join(sim_dir, 'hmm_results')
                Path(sim_hmm_res_dir).mkdir(parents=True, exist_ok=True)

                hmm_res_test_dir = os.path.join(sim_hmm_res_dir, 'test_files_nhmmer')
                Path(hmm_res_test_dir).mkdir(parents=True, exist_ok=True)

                """
                Fasta Creation
                """
                fasta_creation(fasta_file=te_fasta_file, dataframe=sorted_df, train_f=train_folder_f,
                               test_f=test_folder_f, step=seq_step, clean_seq_list=seq_list)
                """
                Alignments
                """
                alignment(train_folder=train_folder_f, alignment_folder=sim_al_dir, debug=main_args.d,
                          cpu_cores=n_cores, redo=redo_align, mode=mafft_mode)

                """
                HMM Build
                """
                hmm_build(alignment_dir=sim_al_dir, hmm_dir=sim_hmm_dir, cpu_cores=n_cores)
                """
                HMM search - complete file
                """
                hmm_search_complete(full_fasta_path=base_path, hmm_dir=sim_hmm_dir, hmm_res_dir=sim_hmm_res_dir,
                                    max_param=nhmmer_max, top_strand=nhmmer_top, cpu_cores=n_cores, redo=redo_search,
                                    train_folder=train_folder_f)
                """
                HMM search - Testing files
                """
                hmm_search_test_files(test_folder=test_folder_f, hmm_dir=sim_hmm_dir,
                                      hmm_test_res_dir=hmm_res_test_dir, max_param=nhmmer_max, top_strand=nhmmer_top,
                                      cpu_cores=n_cores, redo=redo_search, clean_seq_list=seq_list)

                """
                Results and plots
                """
                sim_data_df = results(clean_seq_list=seq_list, train_folder=train_folder_f,
                                      hmm_res_dir=sim_hmm_res_dir,
                                      hmm_test_res_dir=hmm_res_test_dir)
                sim_data_df.to_csv(os.path.join(sim_dir, 'sim_res_max_w_test_.csv'))
                plots(dataframe=sim_data_df, sim_dir=sim_dir, te_name=te_name, hmm_param=hmmsearch_params,
                      aligner=aligner)

                print('Finished analyzing ', te_name)
                print('Creating checkpoint file in ' + str(sim_dir))

                end = timer()
                with open(os.path.join(sim_dir, te_name + '.chk'), 'w') as file:
                    if redo_align:
                        file.write('* Alignment of sequences was redone. *\n')
                    if redo_search:
                        file.write('* nhmmer search was redone. *\n')
                    file.write('Job done for TE super family ' + str(te_name) + '\n')
                    file.write('Elapsed time: ' + str(timedelta(seconds=end - start)) + '\n')
                    file.write('Skipped sequences: \n')
                    for seq in skipped_seqs:
                        file.write(seq + '\n')
                    file.write('Sequence step: ' + str(seq_step) + '\n')
                    file.write('Original fasta file: ' + str(base_path) + '\n')


if __name__ == '__main__':
    desc = 'Pipeline for MSA calculations. Based on the proposed methodology of the thesis developed to achieve the ' \
           'PhD grade. DOI: TBP .'
    parser = argparse.ArgumentParser(description=desc,
                                     epilog='By Camilo Fuentes-Beals. @cakofuentes')
    # Base arguments
    parser.add_argument('--b', '--basedir', required=True, help='Input folder.')
    parser.add_argument('--o', '--output', required=True, help='Output folder.')

    parser.add_argument('--d', '--debug', required=False, action=argparse.BooleanOptionalAction,
                        help='Whether to show or not program logs in execution.')

    parser.add_argument('--l', '--limit', type=int, default=1000, required=False,
                        help='Upper limit of number of sequences. Default 1000')

    parser.add_argument('--s', '--step', type=int, required=False, default=5,
                        help='Default quantity for group of sequences.')
    # cpu arguments.
    parser.add_argument('--n', '--ncore', type=int, required=True,
                        help='Default number of CPU cores to use across the pipeline. For HMMER and MAFFT. '
                             '0 dictates to use all available cpu cores')
    # nhmmer options
    parser.add_argument('--m', '--max', required=False, default=False, action='store_true',
                        help='nhmmer max param. Disabled by default.')
    parser.add_argument('--t', '--ts', required=False, default=False, action='store_true',
                        help='Top strand param for nhmmer. Disabled by default')
    # CD-HIT & options
    parser.add_argument('--c', '--cluster', required=True, action=argparse.BooleanOptionalAction,
                        help='Clustering of sequences, using cd-hit-est.')
    parser.add_argument('--cc', '--identity', required=False, type=float, default=0.90,
                        help='cd-hit-est sequence identity value.')
    parser.add_argument('--cg', '--accurate', required=False, type=int, default=1,
                        help='Accurate or fast mode for cd-hit-est.')
    parser.add_argument('--cm', '--mem', required=False, type=int, default=32000,
                        help='Max memory to be used by cd-hit-est')
    # mafft options
    parser.add_argument('--ma', '--mafft_auto', required=False, default=False, action='store_true',
                        help='Automatic selection for mafft alignment mode. Disabled by default (local alignment).')
    # redo options.
    parser.add_argument('--ra', '--redo-align', required=False, default=False, action='store_true',
                        help='Redo the alignment portion of the analysis.')
    parser.add_argument('--rs', '--redo-search', required=False, default=False, action='store_true',
                        help='Redo the nhmmer (search) portion of the analysis.')

    args = parser.parse_args()

    if args.c and (args.cc is None or args.cg is None):
        parser.error("--c requires --cc and --cg.")

    main(args)

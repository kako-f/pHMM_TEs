import argparse
import os
import sys
import glob
from pathlib import Path
import random
from pyfaidx import Fasta
from tqdm import tqdm
import subprocess
import pandas as pd
import hvplot.pandas  # noqa
import mmap
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from distfit import distfit
import scipy.stats as stats

module_path = str(Path.cwd() / "modules")
if module_path not in sys.path:
    sys.path.append(module_path)
from modules import functions


def simulated_data_creation(fasta_file, data_output, te_name, output_folder):
    """

    :param fasta_file:
    :param data_output:
    :param te_name:
    :param output_folder:
    :return:
    """
    length_dist = {'id': [], 'length': []}
    for seq_id in fasta_file.keys():
        length_dist['id'].append(seq_id)
        length_dist['length'].append(len(fasta_file[seq_id]))
    length_dist_df = pd.DataFrame.from_dict(length_dist)
    length_histogram = length_dist_df['length'].hvplot.hist(width=700, height=400, xlabel='Seq. length',
                                                            ylabel='Quantity', title='TE ' + te_name + ' ' +
                                                                                     str(len(fasta_file.keys())) +
                                                                                     ' sequences.')
    print('Max length of analyzed sequences :', max(length_dist['length']))
    print('Saving plot of length distribution.\n')
    hvplot.save(length_histogram, os.path.join(data_output, te_name + '_length_distribution.html'))

    print('Finding a distribution that fits the length distribution. \n')
    dist = distfit(n_boots=100)
    dist.fit_transform(length_dist_df['length'])

    with open(os.path.join(data_output, te_name + '_fitted_model_info.txt'), 'w') as file:
        for key in dist.model:
            file.write(str(key) + ' : ' + str(dist.model[key]) + '\n')

    print('Saving plot of fitted distribution. \n')
    dist_plot = dist.plot()
    dist_summary = dist.plot_summary()
    dist_summary[0].savefig(os.path.join(data_output, te_name + '_distribution_summary_plot.png'))
    dist_plot[0].savefig(os.path.join(data_output, te_name + '_fitted_distribution_plot.png'))

    print('Getting data from the fitted distribution.\n')
    fitted_data = getattr(stats, dist.model['name'])(dist.model['params'][0], dist.model['params'][1],
                                                     dist.model['params'][2])

    sim_seqs = []
    for seq in fasta_file.keys():
        gc = functions.get_base_frequencies(str(fasta_file[seq]))['G'] + \
             functions.get_base_frequencies(str(fasta_file[seq]))['C']
        base_s = functions.random_dna_seq(length=max(length_dist['length']), p_gc=gc)[0]
        sim_seqs.append(
            functions.fragmentation_with_distr(base_seq=base_s, rep=1, total_seq=1, dist_data=fitted_data)[0][0])

    sim_length_dist = {'length': []}
    for seq in sim_seqs:
        sim_length_dist['length'].append(len(seq))
    sim_length_df = pd.DataFrame.from_dict(sim_length_dist)
    sim_length_df['length'].hvplot.hist(width=700, height=400, xlabel='Seq. length', ylabel='Quantity',
                                        title='Simulated TE ' + te_name + ' ' + str(
                                            len(fasta_file.keys())) + ' sequences.')
    sim_fasta_file = os.path.join(output_folder, 'simulated_' + str(
        len(fasta_file.keys())) + '_' + te_name + '_and_ori_sequences.fasta')
    with open(sim_fasta_file, 'w') as file:
        for i, seq_id in enumerate(fasta_file.keys()):
            file.write('>' + str(seq_id) + '\n')
            file.write(str(fasta_file[seq_id]) + '\n')
            file.write('>simulated_seq_' + str(i + 1) + '\n')
            file.write(str(sim_seqs[i]) + '\n')

    return sim_fasta_file


def fasta_creation(test_fasta_path, output_folder, total_rest_seq, best_fasta_file):
    """
    Function to create shuffled fasta files.
    :param test_fasta_path:
    :param output_folder:
    :param total_rest_seq
    :param best_fasta_file
    :return:
    """
    if total_rest_seq:
        if os.path.exists(os.path.join(output_folder, 'shuffled_fasta_' + str(total_rest_seq) + '_seqs.fasta')):
            pass
        else:
            with open(os.path.join(output_folder, 'shuffled_fasta_' + str(total_rest_seq) + '_seqs.fasta'),
                      'w') as file:
                for id_ in best_fasta_file.keys():
                    file.write('>shuffled_' + str(id_) + '\n')
                    seq_list = list(str(best_fasta_file[id_]))
                    random.shuffle(seq_list)
                    shuffled_seq = ''.join(seq_list)
                    file.write(shuffled_seq + '\n')
                    file.write('>' + str(id_) + '\n')
                    file.write(str(best_fasta_file[id_]) + '\n')
    else:
        for path in tqdm(Path(test_fasta_path).glob('*.fasta'), desc='Shuffling fasta files'):
            fasta_file = Fasta(path)
            n = [int(s) for s in str(path.stem).split('_')[1].split('.') if s.isdigit()][0]
            if os.path.exists(os.path.join(output_folder, 'shuffled_fasta_' + str(n) + '_seqs.fasta')):
                pass
            else:
                with open(os.path.join(output_folder, 'shuffled_fasta_' + str(n) + '_seqs.fasta'), 'w') as file:
                    for id_ in fasta_file.keys():
                        file.write('>shuffled_' + str(id_) + '\n')
                        seq_list = list(str(fasta_file[id_]))
                        random.shuffle(seq_list)
                        shuffled_seq = ''.join(seq_list)
                        file.write(shuffled_seq + '\n')
                        file.write('>' + str(id_) + '\n')
                        file.write(str(fasta_file[id_]) + '\n')


def hmm_search(fasta_path, cpu_cores, max_param, top_strand, hmm_dir, output_path, best_msa, sim):
    if cpu_cores == 0:
        cpu_cores = os.cpu_count()

    for path in tqdm(Path(fasta_path).glob('*.fasta'), desc='Searching on Shuffled Fasta.'):
        base_name = str(path.stem)
        if sim:
            profile_name = 'train_' + str(best_msa) + '_sequences_pw_al_score_profile.hmm'
        else:
            n = [int(s) for s in base_name.split('_')[2].split('.') if s.isdigit()][0]
            profile_name = 'train_' + str(n) + '_sequences_pw_al_score_profile.hmm'

        nhmmer_cmd = 'nhmmer --cpu ' + str(cpu_cores)
        if max_param:
            nhmmer_cmd += ' --max '
        if top_strand:
            nhmmer_cmd += ' --watson '

        nhmmer_cmd += ' -o ' + output_path + '/' + base_name + '.log --tblout ' + output_path + '/' \
                      + base_name + '.out ' + hmm_dir + '/' + profile_name + ' ' + str(path)

        subprocess.run(nhmmer_cmd, shell=True)


def plotting(hmm_res_dir, csv_folder, te, plot_output, sim, ori_file, fasta_file, best_msa):
    roc_curve_data = {
        'group': [],
        'thres': [],
        'tpr': [],
        'fpr': [],
    }
    group_data = {
        'group': [],
        'tn': [],
        'fp': [],
        'fn': [],
        'tp': [],
        'auc': []
    }
    for path in tqdm(Path(hmm_res_dir).glob('*.out'), desc='Plotting'):
        base_name = str(Path(path).stem)
        print(base_name)
        n = [int(s) for s in base_name.split('_')[1].split('.') if s.isdigit()][0]
        print(path)
        df = pd.read_table(path,
                           sep=' ',
                           skipinitialspace=True,
                           comment='#',
                           header=None,
                           usecols=[0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                           names=['target_name', 'hmmfrom', 'hmmto', 'alifrom', 'alito', 'envfrom', 'envto', 'sqlen',
                                  'strand', 'evalue', 'score', 'bias']
                           )
        print(len(df))
        df = df[df['evalue'] < 0.1]
        df.drop_duplicates(subset=['target_name'], inplace=True)
        if ori_file:
            print('Using original fasta file.')
            with open(r'' + str(path), 'rb', 0) as file:
                with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mmap_obj:
                    start = mmap_obj.find(b"Target file")
                    end = mmap_obj.find(b"Option settings")
                    fasta_path = str(mmap_obj[start:end - 3]).split(':')[1].strip().strip('\'')
                    used_fasta_file = Fasta(fasta_path)
        else:
            print('Using best Test fasta file')
            used_fasta_file = Fasta(fasta_file)

        seq_data = {
            'target_name': [],
        }

        if sim:
            sim_data = {
                'target_name': [],
            }
            # adding the rest of the sequences.
            for id_ in used_fasta_file.keys():
                if 'simulated_' in id_:
                    sim_data['target_name'].append(id_)
                else:
                    seq_data['target_name'].append(id_)
            sim_df = pd.DataFrame.from_dict(sim_data)
            seq_df = pd.DataFrame.from_dict(seq_data)
            full_df = pd.concat([df, seq_df, sim_df], ignore_index=True)
        else:
            shuffled_data = {
                'target_name': [],
            }
            # adding the rest of the sequences.
            for id_ in used_fasta_file.keys():
                if 'shuffled_' in id_:
                    shuffled_data['target_name'].append(id_)
                else:
                    seq_data['target_name'].append(id_)
            # creating dataframe with the results and the rest of the sequences
            shuf_df = pd.DataFrame.from_dict(shuffled_data)
            seq_df = pd.DataFrame.from_dict(seq_data)
            full_df = pd.concat([df, seq_df, shuf_df], ignore_index=True)

        full_df.drop_duplicates(subset=['target_name'], inplace=True)
        full_df.reset_index(drop=True, inplace=True)

        # condition - shuffled and simulated means FN sequences.
        if sim:
            con = (full_df['target_name'].str.contains("simulated"))
        else:
            con = (full_df['target_name'].str.contains("shuffled"))

        full_df['te'] = np.where(con, False, True)

        # condition 2 - evalue less than 0.1 are correctly classified as TE.
        # 0.1 also works
        # according to hmmmodel
        con2 = (full_df['evalue'] < 0.1)
        full_df['is_te'] = np.where(con2, True, False)

        # saving csv data
        full_df.to_csv(os.path.join(csv_folder, base_name + '.csv'))

        # calculating things
        cf_mat = confusion_matrix(list(full_df['te']), list(full_df['is_te']))
        tn, fp, fn, tp = cf_mat.ravel()
        fpr, tpr, thresholds = roc_curve(list(full_df['te']), list(full_df['is_te']), pos_label=True)
        # appending data
        roc_curve_data['group'].append(n)
        roc_curve_data['thres'].append(thresholds)
        roc_curve_data['tpr'].append(tpr)
        roc_curve_data['fpr'].append(fpr)

        # more data to be appended
        group_data['group'].append(n)
        group_data['tn'].append(tn)
        group_data['fp'].append(fp)
        group_data['fn'].append(fn)
        group_data['tp'].append(tp)
        group_data['auc'].append(roc_auc_score(list(full_df['te']), list(full_df['is_te'])))
        fdf = pd.DataFrame.from_dict(roc_curve_data)
        fdf.sort_values(by='group', inplace=True)
        fdf.reset_index(drop=True, inplace=True)
        fdf = fdf.explode(['thres', 'tpr', 'fpr'])
        fdf.set_index(['group', 'thres'], inplace=True)

        group_df = pd.DataFrame.from_dict(group_data)
        group_df.sort_values(by='group', inplace=True)
        group_df.reset_index(drop=True, inplace=True)
        group_df.set_index(['group'], inplace=True)
        fdf.to_csv(os.path.join(csv_folder, 'roc_curve_data.csv'))
        group_df.to_csv(os.path.join(csv_folder, 'tn_tp_data.csv'))
        roc_curve_plots = fdf.hvplot.line(x='fpr', y='tpr', groupby='group', width=1000, height=800, padding=0.1,
                                          title='ROC Curve\n' + te + ' data\nAUC: ' + str(
                                              roc_auc_score(list(full_df['te']), list(full_df['is_te']))), )
        if ori_file:
            plot_fname = 'roc_curve_plot_max_' + str(best_msa) + '.html'
        else:
            plot_fname = 'roc_curve_plot_max.html'
        hvplot.save(roc_curve_plots, os.path.join(plot_output, plot_fname))


def main(main_args):
    base_dir = os.path.normpath(main_args.b)
    n_cores = main_args.n

    print('--- Base Options ---')
    print('base dir: ', base_dir)
    print('CPU Cores to use:', n_cores)
    print('debug', main_args.d)
    print('full', main_args.f)
    print('simulation', main_args.s)
    print('original', main_args.o)
    print('--- nhmmer Options ---')
    print('nhmmer max param:', main_args.m)
    print('nhmmer top strand param:', main_args.t)

    for f in glob.glob(base_dir + '/*.chk', recursive=True):
        f = Path(f)
        te_sf = f.stem
        main_folder = f.parent
        print('Working on :', te_sf)
        print('Creating directories.\n')

        hmm_models_path = os.path.join(main_folder, 'hmm_models')

        roc_folder = os.path.join(main_folder, 'roc_analysis')
        Path(roc_folder).mkdir(parents=True, exist_ok=True)

        fasta_path = os.path.join(main_folder, os.path.join('fasta_files', 'test_files'))

        msa_data = pd.read_csv(os.path.join(main_folder, "sim_res_max_w_test_.csv"), index_col=0)
        print('\nFile loaded - ')

        # TODO
        # change selection source

        best_msa_id = msa_data.sort_values(by='test_seqs', ascending=False).index[0]
        analyzed_seqs = msa_data['file_seqs'][best_msa_id]
        rest_seqs = analyzed_seqs - best_msa_id

        if main_args.o:
            """
            Getting info on the original file
            """
            with open(r'' + str(f), 'rb', 0) as file:
                with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mmap_obj:
                    start = mmap_obj.find(b"Original fasta file: ")
                    end = mmap_obj.find(b".fasta")
                    or_fasta_path = str(mmap_obj[start:end]).split(' ')[3].strip('\'') + '.fasta'
                    te_fasta = Fasta(or_fasta_path)
                    print(te_fasta)
        else:
            """
            Using the tested fasta file of the best MSA
            """
            te_fasta = Fasta(os.path.join(fasta_path, 'test_' + str(rest_seqs) + '_sequences_pw_al_score.fasta'))
            print(te_fasta)
            pass

        if main_args.f:
            """
            Shuffled in all fasta files
            """
            shuffled_files_path = os.path.join(roc_folder, 'shuffled_test_fasta_all')
            Path(shuffled_files_path).mkdir(parents=True, exist_ok=True)

            shuffled_hmm_results_path = os.path.join(roc_folder, 'hmm_shuffled_results_all')
            Path(shuffled_hmm_results_path).mkdir(parents=True, exist_ok=True)

            shuffled_csv_data_path = os.path.join(roc_folder, 'csv_data_all')
            Path(shuffled_csv_data_path).mkdir(parents=True, exist_ok=True)
            """
            Creating shuffled fasta.        
            """
            fasta_creation(test_fasta_path=fasta_path, output_folder=shuffled_files_path, total_rest_seq=None,
                           best_fasta_file=None)
            """
            hmm search on shuffled fasta.
            """
            hmm_search(fasta_path=shuffled_files_path, cpu_cores=n_cores, max_param=main_args.m, top_strand=main_args.t,
                       hmm_dir=hmm_models_path, output_path=shuffled_hmm_results_path, best_msa=None, sim=False)
            """
            Plotting and creating CSV
            """
            # hmm_res_dir, csv_folder, te, plot_output, sim, ori_file, fasta_file, best_msa
            plotting(hmm_res_dir=shuffled_hmm_results_path, csv_folder=shuffled_csv_data_path, te=te_sf,
                     plot_output=roc_folder, sim=False, ori_file=True, fasta_file=None, best_msa=None)
        else:
            """
            Using the tested fasta file of the best MSA
            """

            if main_args.s:
                """
                Using simulation based on the best fitted distribution on the data
                """
                sim_csv_data_path = os.path.join(roc_folder, 'simulated_data_csv_best')
                Path(sim_csv_data_path).mkdir(parents=True, exist_ok=True)

                sim_file_path = os.path.join(roc_folder, 'simulated_fasta_best')
                Path(sim_file_path).mkdir(parents=True, exist_ok=True)

                sim_hmm_results_path = os.path.join(roc_folder, 'hmm_simulated_results_best')
                Path(sim_hmm_results_path).mkdir(parents=True, exist_ok=True)

                sim_fasta_file = simulated_data_creation(fasta_file=te_fasta, data_output=roc_folder, te_name=te_sf,
                                                         output_folder=sim_file_path)

                hmm_search(fasta_path=sim_file_path, cpu_cores=n_cores, max_param=main_args.m,
                           top_strand=main_args.t, hmm_dir=hmm_models_path, output_path=sim_hmm_results_path,
                           best_msa=best_msa_id, sim=True)

                plotting(hmm_res_dir=sim_hmm_results_path, csv_folder=sim_csv_data_path, te=te_sf,
                         plot_output=roc_folder, sim=True, ori_file=False, fasta_file=sim_fasta_file,
                         best_msa=best_msa_id)

            else:
                """
                Shuffling the tested fasta file of the best msa
                """
                shuffled_csv_data_path = os.path.join(roc_folder, 'csv_data_best')
                Path(shuffled_csv_data_path).mkdir(parents=True, exist_ok=True)

                shuffled_files_path = os.path.join(roc_folder, 'shuffled_test_fasta_best')
                Path(shuffled_files_path).mkdir(parents=True, exist_ok=True)

                shuffled_hmm_results_path = os.path.join(roc_folder, 'hmm_shuffled_results_best')
                Path(shuffled_hmm_results_path).mkdir(parents=True, exist_ok=True)

                """
                Creating shuffled fasta.        
                """
                fasta_creation(test_fasta_path=fasta_path, output_folder=shuffled_files_path, total_rest_seq=rest_seqs,
                               best_fasta_file=te_fasta)
                """
                hmm search on shuffled fasta.
                """
                hmm_search(fasta_path=shuffled_files_path, cpu_cores=n_cores, max_param=main_args.m,
                           top_strand=main_args.t,
                           hmm_dir=hmm_models_path, output_path=shuffled_hmm_results_path, best_msa=best_msa_id,
                           sim=False)
                """
                Plotting and creating CSV
                """
                plotting(hmm_res_dir=shuffled_hmm_results_path, csv_folder=shuffled_csv_data_path, te=te_sf,
                         plot_output=roc_folder, sim=False, ori_file=main_args.o, fasta_file=False,
                         best_msa=best_msa_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ROC Curve calculations.',
                                     epilog='By Camilo Fuentes-Beals. @cakofuentes')
    # Base arguments
    parser.add_argument('--b', '--basedir', required=True, help='Input folder. Add an * to search an entire folder '
                                                                'with subfolders. Ex. "/path/to/main_folder/*".'
                                                                'Double or single quote needed. ')
    parser.add_argument('--d', '--debug', required=False, action=argparse.BooleanOptionalAction,
                        help='Whether to show or not program logs in execution.')

    parser.add_argument('--f', '--full', required=False, default=False, action='store_true',
                        help='To search for the whole range of available pHMM or only the top one. '
                             'Disabled by default.')
    parser.add_argument('--s', '--sim', required=False, default=False, action='store_true',
                        help='To apply sequence simulation instead of shuffling in the ROC analysis. '
                             'Disabled by default.')
    parser.add_argument('--o', '--or', required=False, default=False, action='store_true',
                        help='Use the original fasta file to do the simulated analysis. Disabled by default.')
    # cpu arguments.
    parser.add_argument('--n', '--ncore', type=int, required=True,
                        help='Default number of CPU cores to use across the pipeline. For HMMER and MAFFT. '
                             '0 dictates to use all available cpu cores')
    # nhmmer options
    parser.add_argument('--m', '--max', required=False, default=False, action='store_true',
                        help='nhmmer max param. Disabled by default.')
    parser.add_argument('--t', '--ts', required=False, default=False, action='store_true',
                        help='Top strand param for nhmmer. Disabled by default')
    args = parser.parse_args()
    main(args)

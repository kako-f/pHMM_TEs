from pathlib import Path
import sys
import argparse
import os
import hvplot.pandas  # noqa
from tqdm import tqdm
import numpy as np
import random
from bokeh.models import HoverTool
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, save, output_file
import parasail
import subprocess
import pandas as pd

module_path = str(Path.cwd() / "modules")
if module_path not in sys.path:
    sys.path.append(module_path)
from modules import functions_v2


def main(main_args):
    max_reps = main_args.r
    base_dir = os.path.normpath(main_args.b)
    gdrive_dir = os.path.normpath(os.path.join(base_dir, 'gdrive'))
    n_seqs = main_args.n
    seq_len = main_args.s
    step = main_args.t
    mutation = main_args.m
    print('base dir =', base_dir)
    print('gdrive dir =', gdrive_dir)
    print('repetitions =', max_reps)
    print('n seqs =', n_seqs)
    print('seq len =', seq_len)
    print('Step =', step)
    print('mutation Type =', mutation)

    mafft_mode = '--localpair'
    strict = False
    n_sel_seqs = int(n_seqs)

    # filename details
    if strict:
        strict_text = 'strict_gap'
        gap_open_penalty = 300
        gap_extension_penalty = 100
    else:
        strict_text = ''
        gap_open_penalty = 10
        gap_extension_penalty = 1

    print('Added information: ')
    print(strict_text)

    repetitions = list(range(1, max_reps, 1))

    # pnt = nucleotide percentages -
    # Provided by the user - if not random percentages would be assigned.
    pnt = []
    if len(pnt) != 0:
        nt_perc = '_'.join([str(elem).replace('.', '') for elem in pnt])
        if len(pnt) < 4:
            raise Exception("Sorry, must be 4 percentages for the 4 nucleotides (ACGT)")
    else:
        nt_perc = 'random'

    if mutation != 'none':
        n_changes = list(range(0, int(seq_len), step))
    else:
        n_changes = [0]

    if pnt:
        base_seq = functions_v2.random_dna_seq(length=seq_len, p_all=pnt, exact=True)
    else:
        base_seq = functions_v2.random_dna_seq(length=seq_len, )
    print('N° Changes: ', n_changes)
    print('Base sequence: ', base_seq[0])
    print('NT percentage: A: ' + str(base_seq[1][0]) + ' C:' + str(base_seq[1][1]) + ' G:' + str(
        base_seq[1][2]) + ' T:' + str(base_seq[1][3]))

    base_folder = os.path.join(base_dir, 'seq_len_' + str(seq_len) + '/n_seq_' + str(n_seqs) + '/pHMM_' + str(
        max(n_changes)) + '_' + mutation + '_test_' + str(nt_perc) + '_nt_perc_n_sel_seqs_' + str(n_sel_seqs))
    gdrive_folder = os.path.join(gdrive_dir, 'pHMM_sim_test/seq_len_' + str(seq_len) + '/n_seq_' + str(
        n_seqs) + '/pHMM_' + str(max(n_changes)) + '_' + mutation + '_test_' + str(
        nt_perc) + '_nt_perc_n_sel_seqs_' + str(n_sel_seqs))
    Path(base_folder).mkdir(parents=True, exist_ok=True)
    Path(gdrive_folder).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(gdrive_folder, 'base_info_' + str(mafft_mode[2:]) + '_' + strict_text + '.txt'),
              'w') as file:
        file.write('Base sequence: ' + base_seq[0] + '\n')
        file.write('NT percentage: \nA: ' + str(base_seq[1][0]) + '\nC: ' + str(base_seq[1][1]) + '\nG: ' + str(
            base_seq[1][2]) + '\nT: ' + str(base_seq[1][3]) + '\n')
        file.write('Seq. len: ' + str(seq_len) + '\n')
        file.write('N° seqs: ' + str(n_seqs) + '\n')
        file.write('Repetitions: ' + str(max(repetitions)) + '\n')
        file.write('MAFFT Mode: ' + mafft_mode[2:] + '\n')
        if strict:
            file.write('Using gap open penalty of ' + str(gap_open_penalty) + '\n')

    for r in tqdm(repetitions, desc='Creating folders by repetition'):
        base_name = 'repetiton_' + str(r)
        rep_folder = os.path.join(base_folder, base_name)
        gdrive_rep_folder = os.path.join(gdrive_folder, base_name)
        Path(rep_folder).mkdir(parents=True, exist_ok=True)
        Path(gdrive_rep_folder).mkdir(parents=True, exist_ok=True)
        for n in n_changes:
            folder_name = 'rep_' + str(r) + '_' + str(n) + '_' + mutation
            work_folder = os.path.join(rep_folder, folder_name)
            gdrive_work_folder = os.path.join(gdrive_rep_folder, folder_name)
            Path(work_folder).mkdir(parents=True, exist_ok=True)
            Path(gdrive_work_folder).mkdir(parents=True, exist_ok=True)
            align_folder = os.path.join(work_folder, 'alignment_' + mafft_mode[2:])
            hmm_model = os.path.join(work_folder, 'hmm_model')
            hmm_res = os.path.join(work_folder, 'hmm_res')
            Path(align_folder).mkdir(parents=True, exist_ok=True)
            Path(hmm_model).mkdir(parents=True, exist_ok=True)
            Path(hmm_res).mkdir(parents=True, exist_ok=True)
    complete_data = {}
    for r in tqdm(repetitions, desc='Analysis by repetition.'):
        base_name = 'repetiton_' + str(r)
        rep_folder = os.path.join(base_folder, base_name)
        gdrive_rep_folder = os.path.join(gdrive_folder, base_name)

        if n_sel_seqs == n_seqs:
            print('Selected sequences is equal to total seqs, skipping sequence selection')
            sel_seqs = n_sel_seqs
        else:
            sel_seqs = random.choices(range(0, n_seqs), k=n_sel_seqs, )
            unique_values, counts = np.unique(sel_seqs, return_counts=True)
            with open(os.path.join(gdrive_rep_folder, 'rep_' + str(r) + '_sel_seq_info.txt'), 'w') as file:
                file.write('Selected sequences and frequences:\n')
                for value, count in zip(unique_values, counts):
                    file.write((f"{value} = {count}" + '\n'))
            # Plot
            p_hist = figure(width=800, height=400, title="Distribution of selected sequences")
            hist, edges = np.histogram(sel_seqs, bins=n_seqs)
            data_source = ColumnDataSource({
                'top': hist,
                'left': edges[:-1],
                'right': edges[1:]
            })
            hover_tool = HoverTool(tooltips=[('Seq. N°', '@top'), ('Left', '@left'), ('Right', '@right')])
            p_hist.quad(bottom=0, source=data_source, fill_color="skyblue", line_color="white",
                        legend_label=str(n_seqs) + ' random samples')
            p_hist.add_tools(hover_tool)
            output_file(filename=os.path.join(gdrive_rep_folder, "hist_sel_seqs_rep_" + str(r) + ".html"),
                        title='Histogram of selected sequences.')
            save(p_hist)
        # Data
        complete_data[r] = {
            'seq_len': [],
            'n_changes': [],
            'new_len': [],
            'align_len': [],
            'median_identity_perc': [],
            'median_gap_perc': [],
            'phmm_hits': [],
            'phmm_len': [],
            'A': [],
            'C': [],
            'G': [],
            'T': []
        }
        for n in tqdm(n_changes, desc='Analysis by mutation.'):
            folder_name = 'rep_' + str(r) + '_' + str(n) + '_' + mutation
            work_folder = os.path.join(rep_folder, folder_name)
            gdrive_work_folder = os.path.join(gdrive_rep_folder, folder_name)
            align_folder = os.path.join(work_folder, 'alignment_' + mafft_mode[2:])
            hmm_model = os.path.join(work_folder, 'hmm_model')
            hmm_res = os.path.join(work_folder, 'hmm_res')
            complete_data[r]['seq_len'].append(len(base_seq[0]))
            complete_data[r]['n_changes'].append(n)
            lenght_dist = {
                'length': []
            }
            temp_a = []
            temp_c = []
            temp_g = []
            temp_t = []

            seq_array = functions_v2.random_mutation(type=mutation, base_seq=base_seq[0], selected_seqs=sel_seqs, rep=1,
                                                     total_seq=n_seqs, n_changes=n, )[0]

            with open(os.path.join(work_folder, folder_name + '_seqs.fasta'), 'w') as file:
                for i, seq in enumerate(seq_array):
                    file.write('>seq_' + str(i) + '\n')
                    file.write(str(seq) + '\n')
                    lenght_dist['length'].append(len(seq))
                    freqs = functions_v2.get_base_frequencies(seq)
                    temp_a.append(freqs['A'])
                    temp_c.append(freqs['C'])
                    temp_g.append(freqs['G'])
                    temp_t.append(freqs['T'])
            complete_data[r]['A'].append(np.average(temp_a))
            complete_data[r]['C'].append(np.average(temp_c))
            complete_data[r]['G'].append(np.average(temp_g))
            complete_data[r]['T'].append(np.average(temp_t))

            len_df = pd.DataFrame.from_dict(lenght_dist)
            len_dist_plot = len_df['length'].hvplot.hist(width=700, height=400, xlabel='Seq. length', ylabel='Quantity',
                                                         title='Length Distribution.')
            hvplot.save(len_dist_plot, os.path.join(work_folder, 'len_dist.html'))
            complete_data[r]['new_len'].append(np.median(len_df['length']))

            identity_perc_dict = []
            seq_gaps_dict = []

            for seq in seq_array:
                for seq_2 in seq_array:
                    # change from nw (global) to sw (local)
                    if mafft_mode == '--globalpair':
                        result = parasail.nw_trace_scan_sat(seq, seq_2, gap_open_penalty, gap_extension_penalty,
                                                            parasail.dnafull)
                    elif mafft_mode == '--localpair':
                        result = parasail.sw_trace_scan_sat(seq, seq_2, gap_open_penalty, gap_extension_penalty,
                                                            parasail.dnafull)
                    identity = (result.traceback.comp.count('|') * 100) / len(result.traceback.ref)
                    gaps = (result.traceback.comp.count(' ') * 100) / len(result.traceback.ref)
                    identity_perc_dict.append(identity)
                    seq_gaps_dict.append(gaps)
            identity_perc_matrix = np.reshape(identity_perc_dict, (len(seq_array), len(seq_array)))
            seq_gaps_matrix = np.reshape(seq_gaps_dict, (len(seq_array), len(seq_array)))
            identity_perc_df = pd.DataFrame(identity_perc_matrix, )
            seq_gaps_df = pd.DataFrame(seq_gaps_matrix, )

            identity_perc_df = identity_perc_df.assign(
                identity_perc_median=identity_perc_df.median(numeric_only=True, axis=1))
            seq_gaps_df = seq_gaps_df.assign(seq_gaps_median=seq_gaps_df.median(numeric_only=True, axis=1))

            identity_perc_df = identity_perc_df.assign(
                identity_perc_variance=identity_perc_df.var(numeric_only=True, axis=1))
            seq_gaps_df = seq_gaps_df.assign(seq_gaps_variance=seq_gaps_df.var(numeric_only=True, axis=1))

            identity_perc_df = identity_perc_df.assign(
                identity_perc_std=identity_perc_df.std(numeric_only=True, axis=1))
            seq_gaps_df = seq_gaps_df.assign(seq_gaps_std=seq_gaps_df.std(numeric_only=True, axis=1))

            identity_perc_df.to_csv(os.path.join(work_folder, 'identity_percentage.csv'), )
            seq_gaps_df.to_csv(os.path.join(work_folder, 'gap_percentage.csv'), )

            # Plot
            data_median = np.median(identity_perc_df['identity_perc_median'])
            data_deviation = np.std(identity_perc_df['identity_perc_median'])
            stdev_range = [data_median - data_deviation, data_median + data_deviation]

            if mutation == 'none':
                p_title = "Distribution of Identity percentages (Median of all vs all) - " + str(n_seqs) + " sequences."
            else:
                p_title = "Distribution of Identity percentages (Median of all vs all) - " + str(
                    n_seqs) + " sequences.\nTesting " + str(mutation) + ". Adding " + str(n) + " random " + str(
                    mutation) + " to random sequences."

            p_dev = figure(width=1000, height=700, x_range=(0, 100), title=p_title)
            hist, edges = np.histogram(identity_perc_df['identity_perc_median'], bins=n_seqs)

            hist_data_source = ColumnDataSource({
                'top': hist,
                'left': edges[:-1],
                'right': edges[1:],
            })
            p_dev.line(x=[data_median, data_median], y=[np.max(hist), 0], line_width=4, line_color='blue',
                       legend_label=f'Average (Mean) Identity %: {data_median}')
            p_dev.quad(bottom=0, top=np.max(hist), left=stdev_range[0], right=stdev_range[1], fill_color='tomato',
                       alpha=0.5, legend_label=f'Standar deviation : {data_deviation}')
            r1 = p_dev.quad(bottom=0, alpha=0.8, source=hist_data_source, fill_color="skyblue", line_color="black",
                            legend_label='Identity Percentage median - ' + str(n_seqs) + ' sequences.')
            hover_tool = HoverTool(tooltips=[('Seq. N°', '@top'), ('Left', '@left'), ('Right', '@right')],
                                   renderers=[r1])

            p_dev.add_tools(hover_tool)
            p_dev.legend.location = "top_left"
            p_dev.legend.click_policy = "hide"
            output_file(filename=os.path.join(gdrive_work_folder,
                                              "deviation_analysis_rep_" + str(r) + "_insertion_" + str(n) + ".html"), )
            save(p_dev)

            complete_data[r]['median_identity_perc'].append(np.median(identity_perc_df['identity_perc_median']))
            complete_data[r]['median_gap_perc'].append(np.median(seq_gaps_df['seq_gaps_median']))

            # Analysis.
            mafft_cmd = 'mafft ' + mafft_mode + ' --thread -1 --reorder ' + work_folder + '/' + folder_name + '_seqs.fasta > ' + align_folder + '/' + folder_name + '_seqs.aln'
            try:
                subprocess.run(mafft_cmd, shell=True, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(e)
            hmmbuild_cmd = 'hmmbuild -o ' + hmm_model + '/' + folder_name + '_seqs.log ' + hmm_model + '/' + folder_name + '_profile.hmm ' + align_folder + '/' + folder_name + '_seqs.aln'
            try:
                subprocess.run(hmmbuild_cmd, shell=True, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(e)
            nhmmer_cmd = 'nhmmer --max -o ' + hmm_res + '/' + folder_name + '_seqs.log --tblout ' + hmm_res + '/' + folder_name + '_seqs.out ' + hmm_model + '/' + folder_name + '_profile.hmm ' + work_folder + '/' + folder_name + '_seqs.fasta'
            try:
                subprocess.run(nhmmer_cmd, shell=True, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(e)

            # phmm and used alignment information

            phmm_detail_data = os.path.join(hmm_model, folder_name + '_seqs.log')
            phmm_detail_df = pd.read_table(phmm_detail_data, sep=' ', skipinitialspace=True, comment='#',
                                           names=['id', 'name', 'nseq', 'alen', 'mlen', 'W', 'eff_nseq', 're/pos',
                                                  'description'])
            mlen = int(phmm_detail_df['mlen'].iloc[0])
            alen = int(phmm_detail_df['alen'].iloc[0])
            complete_data[r]['phmm_len'].append(mlen)
            complete_data[r]['align_len'].append(alen)

            # phmm results

            phmm_res_data = os.path.join(hmm_res, folder_name + '_seqs.out')
            phmm_res_df = pd.read_table(phmm_res_data, sep=' ', skipinitialspace=True, comment='#', header=None,
                                        usecols=[0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                                        names=['target_name', 'hmmfrom', 'hmmto', 'alifrom', 'alito', 'envfrom',
                                               'envto', 'sqlen', 'strand', 'evalue', 'score', 'bias'])
            positive_strand = phmm_res_df[phmm_res_df['strand'] == '+'].copy()
            valid_phmm_res = positive_strand[positive_strand['evalue'] < 0.01].copy()

            complete_data[r]['phmm_hits'].append(len(valid_phmm_res))

    complete_df = pd.concat({k: pd.DataFrame(v) for k, v in complete_data.items()}, axis=0)
    flat_df = complete_df.reset_index()
    flat_df.to_csv(os.path.join(gdrive_folder,
                                mutation + '_data_seq_len_' + str(seq_len) + '_nseqs_' + str(n_seqs) + '_rep_' + str(
                                    max(repetitions)) + '_mafft_mode_' + str(
                                    mafft_mode[2:]) + '_' + strict_text + '.csv'))
    if mutation == 'none':
        plot_source = ColumnDataSource({
            'median_identity_perc': complete_df['median_identity_perc'].to_list(),
            'phmm_hits': complete_df['phmm_hits'].to_list(),
            'colors': np.array([(r, g, 150) for r, g in
                                zip(50 + 2 * np.array(complete_df['median_identity_perc'].to_list()),
                                    30 + 2 * np.array(complete_df['phmm_hits'].to_list()))], dtype="uint8")
        })
        p_title = 'pHMM Hits vs Identity percentage.\nSet of ' + str(n_seqs) + ' sequences.\n Base length: ' + str(
            seq_len) + '\n' + str(max(repetitions)) + ' total repetitions.'
        p_id = figure(width=1000, height=700, x_range=(0, 100), y_range=(0, n_seqs + (0.1 * n_seqs)), title=p_title,
                      x_axis_label='Mean - Identity %', y_axis_label='Mean - pHMM Hits')
        l1 = p_id.scatter(x='median_identity_perc', y='phmm_hits', fill_alpha=0.6, source=plot_source,
                          legend_label='Identity % vs pHMM Hits', line_color='black', radius=0.5, color='colors')
        hover_line = HoverTool(tooltips=[('Mean Identity %', '@median_identity_perc'), ('pHMM hits', '@phmm_hits')],
                               renderers=[l1])
        p_id.add_tools(hover_line)
        p_id.legend.location = "top_left"
        p_id.legend.click_policy = "hide"
        output_file(filename=os.path.join(gdrive_folder, 'plot_data_nseqs_' + str(n_seqs) + '_rep_' + str(
            max(repetitions)) + '_mafft_mode_' + str(mafft_mode[2:]) + '_' + strict_text + '.html'), )
        save(p_id)
    else:
        identity_percentage_data_df = pd.DataFrame()
        for df in flat_df.groupby(by='level_0'):
            inter_df = df[1]
            inter_df.reset_index(drop=True, inplace=True)
            inter_df = inter_df.rename(columns={
                'median_identity_perc': 'median_identity_perc_' + str(df[0]),
                'phmm_hits': 'phmm_hits_' + str(df[0]),
                'n_changes': 'n_changes_' + str(df[0])
            })
            inter_df.sort_values(by='median_identity_perc_' + str(df[0]), ascending=False, inplace=True)
            identity_percentage_data_df = pd.concat([identity_percentage_data_df.reset_index(drop=True),
                                                     inter_df[['median_identity_perc_' + str(df[0])]].reset_index(
                                                         drop=True)], axis=1)
            identity_percentage_data_df = pd.concat([identity_percentage_data_df.reset_index(drop=True),
                                                     inter_df[['phmm_hits_' + str(df[0])]].reset_index(drop=True)],
                                                    axis=1)
            identity_percentage_data_df = pd.concat([identity_percentage_data_df.reset_index(drop=True),
                                                     inter_df[['n_changes_' + str(df[0])]].reset_index(drop=True)],
                                                    axis=1)

        plot_source = ColumnDataSource({
            'median_identity_perc': np.mean(
                identity_percentage_data_df[['median_identity_perc_' + str(x) for x in range(1, max(repetitions) + 1)]],
                axis=1),
            'phmm_hits_mean': np.mean(
                identity_percentage_data_df[['phmm_hits_' + str(x) for x in range(1, max(repetitions) + 1)]], axis=1),
            'phmm_hits_std': np.std(
                identity_percentage_data_df[['phmm_hits_' + str(x) for x in range(1, max(repetitions) + 1)]], axis=1),
            'phmm_hits_left': np.mean(
                identity_percentage_data_df[['phmm_hits_' + str(x) for x in range(1, max(repetitions) + 1)]],
                axis=1) - np.std(
                identity_percentage_data_df[['phmm_hits_' + str(x) for x in range(1, max(repetitions) + 1)]], axis=1),
            'phmm_hits_right': np.mean(
                identity_percentage_data_df[['phmm_hits_' + str(x) for x in range(1, max(repetitions) + 1)]],
                axis=1) + np.std(
                identity_percentage_data_df[['phmm_hits_' + str(x) for x in range(1, max(repetitions) + 1)]], axis=1),
            'mutation_n': identity_percentage_data_df['n_changes_1']
        })
        p_title = 'pHMM Hits vs Identity percentage.\nSet of ' + str(
            n_seqs) + ' modified sequences by ' + mutation + '. Base length: ' + str(seq_len) + '\n' + str(
            max(repetitions)) + ' total repetitions.'
        p_id = figure(width=1000, height=700, x_range=(0, 100), y_range=(0, n_seqs + (0.1 * n_seqs)), title=p_title,
                      x_axis_label='Mean - Identity %', y_axis_label='Mean - pHMM Hits')
        v1 = p_id.varea(x='median_identity_perc', y1='phmm_hits_left', y2='phmm_hits_right', source=plot_source,
                        color='tomato', alpha=0.4, legend_label='SD. area.')
        l1 = p_id.line(x='median_identity_perc', y='phmm_hits_mean', line_width=2, source=plot_source,
                       legend_label='Identity % vs pHMM Hits')
        hover_line = HoverTool(
            tooltips=[('Mean Identity %', '@median_identity_perc'), ('Mean pHMM hits', '@phmm_hits_mean'),
                      ('N° Changes', '@mutation_n')], renderers=[l1])
        hover_area = HoverTool(
            tooltips=[('Mean Identity %', '@median_identity_perc'), ('Min. pHMM hits', '@phmm_hits_left'),
                      ('Max. pHMM hits', '@phmm_hits_right')], renderers=[v1])
        p_id.add_tools(hover_line)
        p_id.add_tools(hover_area)
        p_id.legend.location = "top_left"
        p_id.legend.click_policy = "hide"
        output_file(filename=os.path.join(gdrive_folder,
                                          'plot_' + mutation + '_data_seq_len_' + str(seq_len) + '_nseqs_' + str(
                                              n_seqs) + '_rep_' + str(max(repetitions)) + '_mafft_mode_' + str(
                                              mafft_mode[2:]) + '_' + strict_text + '.html'), )
        save(p_id)

    print(os.path.join(gdrive_folder,
                       'plot_' + mutation + '_data_seq_len_' + str(seq_len) + '_nseqs_' + str(n_seqs) + '_rep_' + str(
                           max(repetitions)) + '_mafft_mode_' + str(mafft_mode[2:]) + '_' + strict_text + '.html'))


if __name__ == '__main__':
    desc = 'alignment of multiple TE sequences.'
    parser = argparse.ArgumentParser(description=desc, epilog='By Camilo Fuentes-Beals. @cakofuentes')
    # Base arguments
    parser.add_argument('--b', '--basedir', required=True, help='Input folder.')
    parser.add_argument('--r', '--repetitions', type=int, default=2, required=False,
                        help='Upper limit of number of repetitions. Default 2.')
    parser.add_argument('--n', '--nseqs', type=int, default=100, required=False,
                        help='Number of sequences. Default 100.')
    parser.add_argument('--s', '--seqlen', type=int, default=100, required=False,
                        help='length of sequences. Default 100.')
    parser.add_argument('--t', '--steps', type=int, default=10, required=False,
                        help='Incremental steps for sequence analysis. Default 10.')
    parser.add_argument('--m', '--mut_type', required=True, help='mutation type.')
    args = parser.parse_args()

    main(args)

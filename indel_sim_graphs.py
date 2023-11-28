from pyfaidx import Fasta
import parasail
import pandas as pd
import numpy as np
import hvplot.pandas  # noqa
import argparse
import glob
import os
from pathlib import Path
import mmap
import subprocess


def main(main_args):
    print(main_args)
    base_dir = os.path.normpath(main_args.b)
    fasta_dir = os.path.normpath(main_args.f)
    sortby = main_args.s
    gap_id = main_args.i
    simulation = main_args.u
    mafft_mode = main_args.ma

    if sortby == 'test':
        sortby = 'test_seqs'
    elif sortby == 'total':
        sortby = 'total_seqs'
    else:
        raise TypeError('Only "test" or "total" are accepted values.')

    if gap_id:
        clean_fasta = []
        fasta_files = []
        for f in glob.glob(fasta_dir + '/**/*clean.fasta', recursive=True):
            clean_fasta.append(f)
        for f in glob.glob(fasta_dir + '/**/*.fasta', recursive=True):
            fasta_files.append(f)
        for f in glob.glob(base_dir + '/**/*.chk', recursive=True):
            file_path = Path(f)
            te_name = file_path.stem
            te_dir = file_path.parent
            msa_process_data = None
            pw_process_data = None
            print('\nWorking on :', te_name)

            try:
                pw_process_data = pd.read_csv(os.path.join(te_dir, te_name + '_pw_al_all_vs_all_score.csv'),
                                              index_col=0)
            except FileNotFoundError as e:
                print("PW calculations file not found. Run the MSA process first. Exception: ", e)
                exit(0)
            else:
                print('PW file loaded.')

            try:
                msa_process_data = pd.read_csv(os.path.join(te_dir, "sim_res_max_w_test_.csv"), index_col=0)
            except FileNotFoundError as e:
                print("MSA calculations file not found. Run the MSA process first. Exception: ", e)
                exit(0)
            else:
                print('MSA file loaded.')

            sorted_pw_df = pw_process_data.loc['column_total'].sort_values(ascending=False).to_frame()
            sorted_res_df = msa_process_data.sort_values(by='test_seqs', ascending=False)
            id_1 = sorted_res_df['base_seqs_1'].iloc[0]

            seq_list = list(sorted_pw_df[:id_1].index)
            try:
                or_fasta = Fasta([s for s in clean_fasta if te_name in s][0])
                print('Clean fasta file found: ', or_fasta)
            except IndexError as e:
                print('Clean Fasta File not found, using normal one.')
                or_fasta = Fasta([s for s in fasta_files if te_name in s][0])
                print('Fasta file found: ', or_fasta)

            base_fasta_path = os.path.join(te_dir, 'top_' + str(id_1) + '_seqs.fasta')
            with open(base_fasta_path, 'w') as file:
                for seq_id in seq_list:
                    file.write('>' + seq_id + '\n')
                    file.write(str(or_fasta[seq_id]) + '\n')

            base_fasta_file = Fasta(base_fasta_path)

            perid_dict = []
            gaps_dict = []
            for seq_id in base_fasta_file.keys():
                for seq_id2 in base_fasta_file.keys():
                    result = parasail.sw_trace(str(base_fasta_file[seq_id]), str(base_fasta_file[seq_id2]), 10, 1,
                                               parasail.dnafull)
                    same = (result.traceback.comp.count('|') * 100) / len(result.traceback.ref)
                    gaps = (result.traceback.comp.count(' ') * 100) / len(result.traceback.ref)
                    perid_dict.append(same)
                    gaps_dict.append(gaps)
            perid_dict = np.array(perid_dict)
            gaps_dict = np.array(gaps_dict)
            perid_dict_matrix = perid_dict.reshape((len(base_fasta_file.keys()), len(base_fasta_file.keys())))
            gapid_dict_matrix = gaps_dict.reshape((len(base_fasta_file.keys()), len(base_fasta_file.keys())))
            perid_df = pd.DataFrame(perid_dict_matrix, index=list(base_fasta_file.keys()),
                                    columns=list(base_fasta_file.keys()))
            gap_df = pd.DataFrame(gapid_dict_matrix, index=list(base_fasta_file.keys()),
                                  columns=list(base_fasta_file.keys()))
            perid_df = perid_df.assign(
                perid_total=perid_df.sum(numeric_only=True, axis=1) / len(base_fasta_file.keys()))
            gap_df = gap_df.assign(gap_total=gap_df.sum(numeric_only=True, axis=1) / len(base_fasta_file.keys()))

            prom_perid = perid_df['perid_total'].hvplot.hist(rot=90, xlim=(0, 100), ylabel='Count', xlabel='% identity',
                                                             width=1000, title='Identity percentage histogram.')
            prom_gap = gap_df['gap_total'].hvplot.hist(rot=90, xlim=(0, 100), ylabel='Count', xlabel='% gaps',
                                                       width=1000,
                                                       title='Gaps percentage histogram.')

            per_id_heatmap = perid_df.loc[:, perid_df.columns != 'perid_total'].hvplot.heatmap(width=1400, height=1400,
                                                                                               rot=90,
                                                                                               title='Identity percentage heatmap.')
            gap_heatmap = gap_df.loc[:, gap_df.columns != 'gap_total'].hvplot.heatmap(width=1400, height=1400, rot=90,
                                                                                      title='Gaps heatmap.')

            hvplot.save(per_id_heatmap, os.path.join(te_dir, 'id_perc_heatmap.html'))
            hvplot.save(gap_heatmap, os.path.join(te_dir, 'gaps_heatmap.html'))
            hvplot.save(prom_perid, os.path.join(te_dir, 'prom_perid.html'))
            hvplot.save(prom_gap, os.path.join(te_dir, 'prom_gap.html'))
    if simulation:
        for f in glob.glob(base_dir + '/**/*.chk', recursive=True):
            file_path = Path(f)
            te_name = file_path.stem
            te_dir = file_path.parent
            msa_process_data = None
            print('\nWorking on :', te_name)

            try:
                msa_process_data = pd.read_csv(os.path.join(te_dir, "sim_res_max_w_test_.csv"), index_col=0)
            except FileNotFoundError as e:
                print("MSA calculations file not found. Run the MSA process first. Exception: ", e)
                exit(0)
            else:
                print('MSA file loaded.')

            msa_process_data = msa_process_data.sort_values(by=sortby, ascending=False)
            msa_id_0 = msa_process_data['base_seqs_0'].iloc[0]
            msa_id_1 = msa_process_data['base_seqs_1'].iloc[0]
            total_seqs = msa_process_data['file_seqs'].iloc[0]
            print('Id. of Best MSA: ', msa_id_1)
            print(total_seqs)

            print('Loading phylogenetic data.')

            tree_file_dir = os.path.join(os.path.join(te_dir, str(msa_id_0) + '_' + str(msa_id_1) + '_data'),
                                         'phylo_inference')

            tree_file = list(Path(tree_file_dir).glob('*.treefile'))[0]
            tree_file_log = list(Path(tree_file_dir).glob('*.iqtree'))[0]

            with open(r'' + str(tree_file_log), 'rb', 0) as file:
                with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mmap_obj:
                    start = mmap_obj.find(b"--alisim")
                    end = mmap_obj.find(b"To mimic ")
                    raw_data = str(mmap_obj[start:end]).split('\"')

                    evo_model_params = raw_data[1]
                    length = raw_data[2].strip()[:-5]
            print('\nEvolution model parameters.')
            print(evo_model_params)
            print(length)

            evo_model_name = evo_model_params.split('{')[0]
            base_path = os.path.join(te_dir, 'simulations/' + str(msa_id_0) + '_' + str(msa_id_1) + '_data')
            sim_path = os.path.join(base_path, 'res_' + evo_model_name + '/')
            hmm_res_folder = os.path.join(base_path, 'hmm_res_' + evo_model_name + '/')
            hmm_res_folder_base = os.path.join(os.path.join(base_path, 'hmm_res_' + evo_model_name), 'base_res')
            hmm_model_folder = os.path.join(base_path, 'hmm_model_' + evo_model_name + '/')

            Path(sim_path).mkdir(parents=True, exist_ok=True)
            Path(hmm_res_folder).mkdir(parents=True, exist_ok=True)
            Path(hmm_res_folder_base).mkdir(parents=True, exist_ok=True)
            Path(hmm_model_folder).mkdir(parents=True, exist_ok=True)

            print('\nFolders.')

            print(sim_path)
            print(hmm_res_folder)
            print(hmm_model_folder)

            print('\nAnalysis process.')
            # IQTREE Normal
            alisim_cmd = 'iqtree2 --alisim ' + sim_path + evo_model_name + '_sim -m "' + evo_model_params \
                         + '" -t ' + str(tree_file) + ' --out-format fasta ' + str(length) + ' --seed 123 --seqtype DNA'
            try:
                print('IQTREE - ALISIM cmd: ', alisim_cmd)
                if os.path.exists(sim_path + evo_model_name + '_sim.fa'):
                    print('File exist  Skipping')
                else:
                    subprocess.run(alisim_cmd, shell=True, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(e)

            # MAFFT
            mafft_cmd = 'mafft '
            if mafft_mode:
                mafft_cmd += ' --auto'
            else:
                mafft_cmd += ' --localpair'
            mafft_cmd += ' --reorder ' + sim_path + evo_model_name + '_sim.fa > ' + \
                         sim_path + evo_model_name + '.aln'
            try:
                print('MAFFT cmd: ', mafft_cmd)
                if os.path.exists(sim_path + evo_model_name + '.aln'):
                    print('File exist - Skipping')
                else:
                    subprocess.run(mafft_cmd, shell=True, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(e)

            # hmmbuild
            hmmbuild_cmd = 'hmmbuild -o ' + hmm_model_folder + evo_model_name + '_sim.log ' + \
                           hmm_model_folder + evo_model_name + '_profile.hmm ' + sim_path + evo_model_name + '.aln'
            try:
                print('hmmbuild cmd: ', hmmbuild_cmd)
                if os.path.exists(hmm_model_folder + evo_model_name + '_sim.log'):
                    print('File exist - Skipping')
                else:
                    subprocess.run(hmmbuild_cmd, shell=True, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(e)

            # nhmmer
            nhmmer_cmd = 'nhmmer --max -o ' + hmm_res_folder_base + '/' + evo_model_name \
                         + '_sim.log --tblout ' + hmm_res_folder_base + '/' + evo_model_name + '_sim.out ' + \
                         hmm_model_folder + evo_model_name + '_profile.hmm ' + sim_path + evo_model_name + '_sim.fa'
            try:
                print('nhmmer cmd: ', nhmmer_cmd)
                if os.path.exists(hmm_res_folder_base + '/' + evo_model_name + '_sim.log'):
                    print('File exist - Skipping')
                else:
                    subprocess.run(nhmmer_cmd, shell=True, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(e)

            print('\nIndel simulation.')
            for ir in np.arange(0.01, 0.11, 0.01):
                for dr in np.arange(0.01, 0.11, 0.01):
                    alisim_cmd = 'iqtree2 --alisim ' + sim_path + evo_model_name + '_ir_' + str(ir) + '_dr_' + str(
                        dr) + ' -m "' + evo_model_params + '" --indel ' + str(ir) + ',' + str(
                        dr) + ' -t ' + str(tree_file) + ' --out-format fasta --seed 123 --seqtype DNA'
                    try:
                        # print(alisim_cmd)
                        subprocess.run(alisim_cmd, shell=True, check=True, capture_output=True)
                    except subprocess.CalledProcessError as e:
                        print(e)
            # Indel nhmmer
            files = sorted(glob.glob(sim_path + '/*.unaligned.fa'),
                           key=lambda x: float(os.path.basename(x).split('_')[2]))
            for fpath in files:
                fname = Path(fpath).stem
                nhmmer_cmd = 'nhmmer --max -o ' + hmm_res_folder + fname + '.log --tblout ' + hmm_res_folder \
                             + fname + '.out ' + hmm_model_folder + evo_model_name + '_profile.hmm ' + str(fpath)

                try:
                    if os.path.exists(hmm_res_folder + '/' + fname + '.log'):
                        print('File exist - Skipping')
                    else:
                        subprocess.run(nhmmer_cmd, shell=True, check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    print(e)

            hmm_results = {}
            hmm_res_base_data = os.path.join(hmm_res_folder_base, evo_model_name + '_sim.out')
            df = pd.read_table(hmm_res_base_data,
                               sep=' ',
                               skipinitialspace=True,
                               comment='#',
                               header=None,
                               usecols=[0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                               names=['target_name', 'hmmfrom', 'hmmto', 'alifrom', 'alito', 'envfrom', 'envto',
                                      'sqlen',
                                      'strand', 'evalue', 'score', 'bias']
                               )
            hmm_results['base'] = {
                'ir': 0,
                'dr': 0,
                'total': len(df),
                'total_eval': len(df[df['evalue'] < 0.01])
            }
            files = sorted(glob.glob(hmm_res_folder + '/*.out'),
                           key=lambda x: float(os.path.basename(x).split('_')[2]))  # 3
            for fpath in files:
                ir_ = np.round(float(os.path.basename(fpath).split('_')[2]), decimals=3)  # 3
                dr_ = np.round(float(os.path.basename(fpath).split('_')[4].split('.unaligned')[0]), decimals=3)  # 5
                df = pd.read_table(fpath,
                                   sep=' ',
                                   skipinitialspace=True,
                                   comment='#',
                                   header=None,
                                   usecols=[0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                                   names=['target_name', 'hmmfrom', 'hmmto', 'alifrom', 'alito', 'envfrom', 'envto',
                                          'sqlen',
                                          'strand', 'evalue', 'score', 'bias']
                                   )
                total_hits = len(df)
                eval_hits = len(df[df['evalue'] < 0.01])
                hmm_results[str(ir_) + '_' + str(dr_)] = {
                    'ir': ir_,
                    'dr': dr_,
                    'total': len(df),
                    'total_eval': len(df[df['evalue'] < 0.01])
                }
            hmm_res_df = pd.DataFrame.from_dict(hmm_results, orient='index')
            indel_plot = hmm_res_df.hvplot.line(y='total_eval', rot=90, height=500, width=1500,
                                                ylabel='Total hits by evalue', xlabel='Indel rate',
                                                title='Indel simulation.')
            hvplot.save(indel_plot, os.path.join(base_path, 'indel_plot.html'))

            sorted_hmm_res_df = hmm_res_df.sort_values(by='total_eval', ascending=False).copy()

            indel_plot_sorted = sorted_hmm_res_df.hvplot.line(y='total_eval', rot=90, height=500, width=1500,
                                                              ylabel='Total hits by evalue', xlabel='Indel rate',
                                                              title='Indel simulation. Sorted by pHMM result.')
            hvplot.save(indel_plot_sorted, os.path.join(base_path, 'indel_plot_sorted.html'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyzing Gaps and insertion percentages.',
                                     epilog='By Camilo Fuentes-Beals. @cakofuentes')
    # Base arguments
    parser.add_argument('--f', '--fastadir', required=True, help='Fasta files folder. Should be the folder where the '
                                                                 'fasta files are located.')
    parser.add_argument('--b', '--basedir', required=True,
                        help='Input folder. Should be the output folder of the MSA Calculations.')
    parser.add_argument('--s', '--sortby', required=False, default='test',
                        help='Whether to sort by test files or total file. Regarding pHMM results. '
                             'Default: test. Valid options: test or total')
    # Module selection.
    parser.add_argument('--i', '--idgap', required=False, default=False, action='store_true',
                        help='Run the Identity and gaps, graphs module.')
    parser.add_argument('--u', '--sim', required=False, default=False, action='store_true',
                        help='Run the simulation (indel) graphs module.')
    parser.add_argument('--d', '--debug', required=False, action=argparse.BooleanOptionalAction,
                        help='Whether to show or not program logs in execution.')
    # mafft options
    parser.add_argument('--ma', '--mafft_auto', required=False, default=False, action='store_true',
                        help='Automatic selection for MAFFT alignment mode. Disabled by default (local alignment).')
    args = parser.parse_args()

    main(args)

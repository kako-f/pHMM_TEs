import argparse
import glob
import mmap
import os
import shutil
import subprocess
import uuid
import sys
from pathlib import Path
import pandas as pd
from timeit import default_timer as timer
from datetime import timedelta


def asr_hmm_search(work_dir, ncores, max_param, top_strand, asr_msa, phmm_path, asr_fasta_file, original_fasta,
                   test_fasta):
    if ncores == 0:
        ncores = os.cpu_count()
    print('---' * 2)
    print('ASR HMM search.')
    print('\nCreating directories.')
    hmm_res_dir = os.path.join(work_dir, 'asr_hmm_results')
    Path(hmm_res_dir).mkdir(parents=True, exist_ok=True)

    base_name = str(Path(asr_msa).stem)
    print('\nSearching on ASR Fasta file.')
    nhmmer_cmd = 'nhmmer --cpu ' + str(ncores)
    if max_param:
        nhmmer_cmd += ' --max'
    if top_strand:
        nhmmer_cmd += ' --watson'

    nhmmer_cmd += ' -o ' + os.path.join(hmm_res_dir, base_name + '.log') + ' --tblout ' + os.path.join(
        hmm_res_dir, base_name + '.out ') + ' ' + phmm_path + ' ' + asr_fasta_file
    print('nhmmer command: ', nhmmer_cmd)
    if os.path.exists(os.path.join(hmm_res_dir, base_name + '.log')):
        print('File exist - Skipping')
    else:
        subprocess.run(nhmmer_cmd, shell=True)

    print('\nSearching on Original Fasta file.')
    nhmmer_cmd_or = 'nhmmer --cpu ' + str(ncores)
    if max_param:
        nhmmer_cmd_or += ' --max'
    if top_strand:
        nhmmer_cmd_or += ' --watson'

    nhmmer_cmd_or += ' -o ' + os.path.join(hmm_res_dir, base_name + '_or_fasta.log') + ' --tblout ' + os.path.join(
        hmm_res_dir, base_name + '_or_fasta.out') + ' ' + phmm_path + ' ' + original_fasta
    print('nhmmer command: ', nhmmer_cmd_or)
    if os.path.exists(os.path.join(hmm_res_dir, base_name + '_or_fasta.log')):
        print('File exist - Skipping')
    else:
        subprocess.run(nhmmer_cmd_or, shell=True)

    print('\nSearching on Test Fasta file.')
    nhmmer_cmd_test = 'nhmmer --cpu ' + str(ncores)
    if max_param:
        nhmmer_cmd_test += ' --max'
    if top_strand:
        nhmmer_cmd_test += ' --watson'
    nhmmer_cmd_test += ' -o ' + os.path.join(hmm_res_dir, base_name + '_test_fasta.log') + ' --tblout ' + os.path.join(
        hmm_res_dir, base_name + '_test_fasta.out') + ' ' + phmm_path + ' ' + test_fasta
    print('nhmmer command: ', nhmmer_cmd_test)
    if os.path.exists(os.path.join(hmm_res_dir, base_name + '_test_fasta.log')):
        print('File exist - Skipping')
    else:
        subprocess.run(nhmmer_cmd_test, shell=True)


def asr_hmm_build(work_dir, asr_msa, ncores):
    if ncores == 0:
        ncores = os.cpu_count()

    print('---' * 2)
    print('ASR HMM build.')
    print('\nCreating directories.')

    hmm_dir = os.path.join(work_dir, 'asr_hmm_model')
    Path(hmm_dir).mkdir(parents=True, exist_ok=True)

    base_name = str(Path(asr_msa).stem)
    hmm_path = os.path.join(hmm_dir, base_name + '_profile.hmm')
    hmmbuild_cmd = 'hmmbuild --cpu ' + str(ncores) + ' -o ' + \
                   os.path.join(hmm_dir, base_name + '.log') + ' ' + hmm_path + ' ' + asr_msa
    print('hmmbuild command: ', hmmbuild_cmd)
    if os.path.exists(os.path.join(hmm_dir, base_name + '.log')):
        print('File exist - Skipping')
    else:
        subprocess.run(hmmbuild_cmd, shell=True)
    print('pHMM built successfully.')

    return hmm_path


def asr_analysis(work_dir_asr, asr_output_name, msa_id, debug, ncores, base_dir, f_id):
    print('---' * 2)
    print('Analyzing ASR data.')
    asr_state_path = os.path.join(work_dir_asr, asr_output_name + '.state')
    asr_state_df = pd.read_csv(asr_state_path, sep='\t', comment='#')
    align_length = asr_state_df['Site'].max()
    nodes = asr_state_df['Node'].unique().tolist()

    with open(os.path.join(base_dir, 'phmm_asr_execution_' + f_id + '.log'), 'a+') as file:
        file.write('ASR information.\n')
        print('ASR information.\n')
        file.write('ASR Alignment length: ' + str(align_length) + '\n')
        print('ASR Alignment length: ' + str(align_length) + '\n')
        file.write('ASR Nodes: ' + str(len(nodes)) + '\n')
        print('ASR Nodes: ' + str(len(nodes)) + '\n')

    print('\nWriting ASR Fasta sequences.')
    fasta_dir = os.path.join(work_dir_asr, 'asr_fasta')
    Path(fasta_dir).mkdir(parents=True, exist_ok=True)

    asr_fasta_file = os.path.join(fasta_dir, 'msa_' + str(msa_id[0]) + '_' + str(msa_id[1]) + '_asr.fasta')
    print('FASTA file will be saved in: ', asr_fasta_file)
    with open(asr_fasta_file, 'w') as f:
        for i, node in enumerate(nodes):
            start = align_length * i
            end = align_length * (i + 1)
            node_seq = ''.join(asr_state_df[start:end]['State'].values)
            node_seq = node_seq.replace("-", "")
            f.write('>' + node + '\n')
            f.write(node_seq + '\n')

    # Alignment
    # Not necessary to do local alignment because we know that these are full sequences.
    print('---' * 2)
    print('Aligning ASR sequences.')
    aln_dir = os.path.join(fasta_dir, 'asr_align')
    Path(aln_dir).mkdir(parents=True, exist_ok=True)
    asr_fasta_msa = os.path.join(aln_dir, 'msa_' + str(msa_id[0]) + '_' + str(msa_id[1]) + '_asr.aln')
    mafft_cmd = 'mafft'
    if not debug:
        mafft_cmd += ' --quiet'
    if ncores == 0:
        ncores = '-1'
    mafft_cmd += ' --thread ' + str(ncores) + ' --reorder ' + asr_fasta_file + ' > ' + asr_fasta_msa
    print('MAFFT command: ', mafft_cmd)
    if os.path.exists(asr_fasta_msa):
        if os.stat(asr_fasta_msa).st_size > 0:
            print('File exist. Skipping.')
        else:
            subprocess.run(mafft_cmd, shell=True)
    else:
        subprocess.run(mafft_cmd, shell=True)

    return asr_fasta_file, asr_fasta_msa


def iqtree_asr(work_dir, log, tree_file, msa_id, msa_path, ncores, debug):
    asr_dir = os.path.join(work_dir, 'asr_data')
    Path(asr_dir).mkdir(parents=True, exist_ok=True)
    if ncores == 0:
        ncores = 'AUTO'
    with open(r'' + log, 'rb', 0) as file:
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mmap_obj:
            start = mmap_obj.find(b"Best-fit model according to BIC:")
            end = mmap_obj.find(b"List")
            evo_model = str(mmap_obj[start:end - 2]).split(':')[1].strip().strip('\'')
    print('---' * 2)
    print('\nAncestral Sequence Reconstruction')

    iqtree_output_name = 'phylo_inf_msa_' + str(msa_id[0]) + '_' + str(msa_id[1])
    iqtree_output_name_asr = iqtree_output_name + '_asr'

    iqtree_asr_cmd = 'iqtree2 '
    if not debug:
        iqtree_asr_cmd += '-quiet '
    iqtree_asr_cmd += ' -s ' + msa_path + ' -T ' + str(ncores) + ' -asr -te ' + str(
        tree_file) + ' -m ' + evo_model + ' -pre ' + os.path.join(asr_dir, iqtree_output_name_asr)
    print('IQTREE ASR command: ', iqtree_asr_cmd)
    subprocess.run(iqtree_asr_cmd, shell=True)

    return asr_dir, iqtree_output_name_asr


def iqtree_process(work_dir, ncores, msa_id, alrt, ufboot, pers, nstop, msa_path, debug):
    phylo_dir = os.path.join(work_dir, 'phylo_inference')
    Path(phylo_dir).mkdir(parents=True, exist_ok=True)

    if ncores == 0:
        ncores = 'AUTO'

    print('---' * 2)
    print('Phylogenetic inference directory: ', work_dir)
    iqtree_output_name = 'phylo_inf_msa_' + str(msa_id[0]) + '_' + str(msa_id[1])
    if alrt:
        iqtree_output_name += '_alrt'
    if ufboot:
        iqtree_output_name += '_ufboot'
    if pers:
        iqtree_output_name += '_pers'
    if nstop:
        iqtree_output_name += '_nstop'

    iqtree_cmd = 'iqtree2 -s ' + msa_path + ' -T ' + str(ncores) + ' -st DNA'
    if not debug:
        iqtree_cmd += ' -quiet '

    if alrt:
        iqtree_cmd += ' -alrt ' + str(alrt)
    if ufboot:
        iqtree_cmd += ' -B ' + str(ufboot)
    if pers:
        iqtree_cmd += ' -pers ' + str(pers)
    if nstop:
        iqtree_cmd += ' -nstop ' + str(nstop)

    iqtree_cmd += ' -pre ' + os.path.join(phylo_dir, iqtree_output_name)
    print('IQTREE command: ', iqtree_cmd)

    subprocess.run(iqtree_cmd, shell=True)

    iqtree_log = os.path.join(phylo_dir, iqtree_output_name + '.iqtree')
    tree_file = os.path.join(phylo_dir, iqtree_output_name + '.treefile')

    rooted_dir = os.path.join(phylo_dir, 'rooted')
    Path(rooted_dir).mkdir(parents=True, exist_ok=True)

    iqtree_root_cmd = 'iqtree2 -m 12.12 -s ' + msa_path + ' -T ' + str(ncores)

    if not debug:
        iqtree_root_cmd += ' -quiet '

    iqtree_root_cmd += ' -te ' + str(tree_file) + ' -pre ' + str(rooted_dir) + '/' + iqtree_output_name + '_rooted'
    print('IQTREE root command: ', iqtree_root_cmd)

    subprocess.run(iqtree_root_cmd, shell=True)
    rooted_tree_file = os.path.join(rooted_dir, iqtree_output_name + '_rooted.treefile')
    return iqtree_log, rooted_tree_file


def main(main_args):
    print(main_args)
    base_dir = os.path.normpath(main_args.b)
    n_cores = main_args.n
    nhmmer_max = main_args.m
    nhmmer_top = main_args.t
    iqtree_alrt = main_args.a
    iqtree_ub = main_args.ub
    iqtree_ns = main_args.ns
    iqtree_pers = main_args.p
    sortby = main_args.s

    if sortby == 'test':
        sortby = 'test_seqs'
    elif sortby == 'total':
        sortby = 'total_seqs'
    else:
        raise TypeError('Only "test" or "total" are accepted values.')

    print('--- Base Options ---')
    print('* Base dir: ', base_dir)
    print('* CPU Cores to use:', n_cores)
    print('* Sort by:', sortby)
    print('* Verbose/Debug: ', main_args.d)
    print('--- nhmmer Options ---')
    print('nhmmer max param:', nhmmer_max)
    print('nhmmer top strand param:', nhmmer_top)
    print('--- IQTREE Options ---')
    print('alrt: ', iqtree_alrt)
    print('ufboot: ', iqtree_ub)
    print('nstop: ', iqtree_ns)
    print('pers: ', iqtree_pers)
    print('----------------------\n')

    f_id = str(uuid.uuid4())
    with open(os.path.join(base_dir, 'phmm_asr_execution_' + f_id + '.log'), 'w+') as file:
        file.write('Command: ' + f"{' '.join(sys.argv)}")
        file.write('\n--- Base Options ---')
        file.write('\n* Base dir: ' + str(base_dir))
        file.write('\n* CPU Cores to use:' + str(n_cores))
        file.write('\n* Verbose/Debug: ' + str(main_args.d))
        file.write('\n--- nhmmer Options ---')
        file.write('\nnhmmer max param:' + str(nhmmer_max))
        file.write('\nnhmmer top strand param:' + str(nhmmer_top))
        file.write('\n--- IQTREE Options ---')
        file.write('\nalrt: ' + str(iqtree_alrt))
        file.write('\nufboot: ' + str(iqtree_ub))
        file.write('\nnstop: ' + str(iqtree_ns))
        file.write('\npers: ' + str(iqtree_pers))
        file.write('\n----------------------\n')

    start_t = timer()
    for f in glob.glob(base_dir + '/**/*.chk', recursive=True):

        file_path = Path(f)
        te_name = file_path.stem
        te_dir = file_path.parent
        msa_process_data = None
        print('Working on :', te_name)

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

        print('\n==Creating folder structure.==')
        print('Main Folder')
        top_dir = os.path.join(te_dir, str(msa_id_0) + '_' + str(msa_id_1) + '_data')
        try:
            Path(top_dir).mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print('Error: ', exc)
        else:
            print('Folder created - ' + str(top_dir))

        print('MSA folder.')
        best_msa_dir = os.path.join(top_dir, 'alignment_files')
        try:
            Path(best_msa_dir).mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print('Error: ', exc)
        else:
            print('Folder created - ' + str(best_msa_dir))

        print('pHMM and results folder.')
        best_phmm_dir = os.path.join(top_dir, 'hmm_models')
        best_phmm_res_dir = os.path.join(top_dir, 'hmm_results')
        phhm_res_dir_test = os.path.join(best_phmm_res_dir, 'test_files_nhmmer')
        try:
            Path(best_phmm_dir).mkdir(parents=True, exist_ok=True)
            Path(best_phmm_res_dir).mkdir(parents=True, exist_ok=True)
            Path(phhm_res_dir_test).mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print('Error: ', exc)
        else:
            print('Folders created - ' + str(best_phmm_dir) + ' and ' + str(best_phmm_res_dir) + '.')

        print('\nObtaining data.')
        # Expected files and folders paths.
        # MSA
        msa_folder = os.path.join(te_dir, 'alignment_files')
        if msa_id_0 == 0:
            msa_file_name = os.path.join('base_align', str(msa_id_0) + '_' + str(
                msa_id_1) + '_train_sequences_pw_al_score.aln')
        else:
            msa_file_name = str(msa_id_0) + '_' + str(msa_id_1) + '_train_sequences_pw_al_score_addfragments.aln'
        msa_path = os.path.join(msa_folder, msa_file_name)
        print('Best MSA path: ', msa_path)
        # pHMM
        phmm_folder = os.path.join(te_dir, 'hmm_models')
        phmm_file_name_log = str(msa_id_0) + '_' + str(msa_id_1) + '_train_sequences_pw_al_score_addfragments.log'
        phmm_file_name_hmm = str(msa_id_0) + '_' + str(
            msa_id_1) + '_train_sequences_pw_al_score_addfragments_profile.hmm'
        phmm_fname_log_path = os.path.join(phmm_folder, phmm_file_name_log)
        phmm_fname_hmm_path = os.path.join(phmm_folder, phmm_file_name_hmm)
        # Results
        phmm_results_folder = os.path.join(te_dir, 'hmm_results')
        phmm_res_sub_folder = os.path.join(phmm_results_folder, 'hmm_res_test_files')
        phmm_results_fname_log = str(msa_id_0) + '_' + str(
            msa_id_1) + '_train_sequences_pw_al_score_addfragments_profile.log'
        phmm_results_fname_out = str(msa_id_0) + '_' + str(
            msa_id_1) + '_train_sequences_pw_al_score_addfragments_profile.out'

        phmm_res_sub_fname_log = str(msa_id_1) + '_vs_' + str(int(total_seqs) - int(msa_id_1)) + '_test_sequences.log'
        phmm_res_sub_fname_out = str(msa_id_1) + '_vs_' + str(int(total_seqs) - int(msa_id_1)) + '_test_sequences.out'

        phmm_res_fname_log_path = os.path.join(phmm_results_folder, phmm_results_fname_log)
        phmm_res_fname_out_path = os.path.join(phmm_results_folder, phmm_results_fname_out)
        phmm_res_sub_fname_log_path = os.path.join(phmm_res_sub_folder, phmm_res_sub_fname_log)
        phmm_res_sub_fname_out_path = os.path.join(phmm_res_sub_folder, phmm_res_sub_fname_out)

        test_fasta_path = os.path.join(os.path.join(os.path.join(te_dir, 'fasta_files'), 'test_files'),
                                       str(int(total_seqs) - int(msa_id_1)) + '_test_sequences_pw_al_score.fasta')
        print('Copying MSA data.')
        best_msa_path = os.path.join(best_msa_dir, msa_file_name)
        try:
            shutil.copy2(msa_path, best_msa_path)
        except Exception as exc:
            print('Error: ', exc)
        else:
            print('MSA files copied successfully - ')

        print('Copying pHMM data.')
        try:
            shutil.copy2(phmm_fname_log_path, os.path.join(best_phmm_dir, phmm_file_name_log))
            shutil.copy2(phmm_fname_hmm_path, os.path.join(best_phmm_dir, phmm_file_name_hmm))

            shutil.copy2(phmm_res_fname_log_path, os.path.join(best_phmm_res_dir, phmm_results_fname_log))
            shutil.copy2(phmm_res_fname_out_path, os.path.join(best_phmm_res_dir, phmm_results_fname_out))

            shutil.copy2(phmm_res_sub_fname_log_path, os.path.join(phhm_res_dir_test, phmm_res_sub_fname_log))
            shutil.copy2(phmm_res_sub_fname_out_path, os.path.join(phhm_res_dir_test, phmm_res_sub_fname_out))

        except Exception as exc:
            print('Error: ', exc)
        else:
            print('Files copied successfully - ')

        with open(r'' + str(f), 'rb', 0) as file:
            with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mmap_obj:
                start = mmap_obj.find(b"Original fasta file: ")
                end = mmap_obj.find(b".fasta")
                or_fasta_path = str(mmap_obj[start:end]).split(' ')[3].strip('\'') + '.fasta'

        iqtree_log, tree_file = iqtree_process(work_dir=top_dir, ncores=n_cores, msa_id=[msa_id_0, msa_id_1],
                                               alrt=iqtree_alrt, ufboot=iqtree_ub, nstop=iqtree_ns, pers=iqtree_pers,
                                               msa_path=msa_path, debug=main_args.d)
        asr_dir, asr_output_name = iqtree_asr(work_dir=top_dir, log=iqtree_log, tree_file=tree_file,
                                              msa_id=[msa_id_0, msa_id_1],
                                              msa_path=msa_path, ncores=n_cores, debug=main_args.d)

        asr_file, asr_msa = asr_analysis(work_dir_asr=asr_dir, asr_output_name=asr_output_name,
                                         msa_id=[msa_id_0, msa_id_1], debug=main_args.d, ncores=n_cores,
                                         base_dir=base_dir, f_id=f_id)

        asr_phmm = asr_hmm_build(work_dir=top_dir, asr_msa=asr_msa, ncores=n_cores)
        asr_hmm_search(work_dir=top_dir, ncores=n_cores, max_param=nhmmer_max, top_strand=nhmmer_top,
                       phmm_path=asr_phmm, asr_fasta_file=asr_file, original_fasta=or_fasta_path, asr_msa=asr_msa,
                       test_fasta=test_fasta_path)

        end_t = timer()
        print('\nDone.')
        with open(os.path.join(base_dir, 'phmm_asr_execution_' + f_id + '.log'), 'a+') as file:
            file.write('Id. of Best MSA: ' + str(msa_id_1) + '_vs_' + str(msa_id_1))
            file.write('Job done for MSA with id: ' + str(msa_id_1) + '_vs_' + str(msa_id_1) + '\n')
            file.write('Elapsed time: ' + str(timedelta(seconds=end_t - start_t)) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline for pHMM creation and ASR.',
                                     epilog='By Camilo Fuentes-Beals. @cakofuentes')
    # Base arguments
    parser.add_argument('--b', '--basedir', required=True,
                        help='Input folder. Should be the output folder of the MSA Calculations.')
    parser.add_argument('--d', '--debug', required=False, action=argparse.BooleanOptionalAction,
                        help='Whether to show or not program logs in execution.')
    parser.add_argument('--s', '--sortby', required=False, default='test',
                        help='Whether to sort by test files or total file. Regarding pHMM results. '
                             'Default: test. Valid options: test or total')

    # cpu arguments.
    parser.add_argument('--n', '--ncore', type=int, required=False, default=0,
                        help='Default number of CPU cores to use across the pipeline.'
                             '0 dictates to use all available cpu cores')
    # nhmmer options
    parser.add_argument('--m', '--max', required=False, default=False, action='store_true',
                        help='nhmmer max param. Disabled by default.')
    parser.add_argument('--t', '--ts', required=False, default=False, action='store_true',
                        help='Top strand param for nhmmer. Disabled by default')
    # iqtree options
    parser.add_argument('--a', '--alrt', type=int, required=False, default=1000,
                        help='Specify number of replicates (>=1000) to perform SH-like approximate likelihood ratio '
                             'test (SH-aLRT) in IQTREE. If number of replicates is set to 0 (-alrt 0), '
                             'then the parametric aLRT test is performed, instead of SH-aLRT. Default: 1000')
    parser.add_argument('--ub', '--ufboot', type=int, required=False, default=1000,
                        help='Replicates for the ultrafast bootstrap (>=1000) for IQTREE. Default: 1000')
    parser.add_argument('--ns', '--nstop', type=int, required=False, default=100,
                        help='Specify number of unsuccessful iterations to stop in IQTREE. Default: 100')
    parser.add_argument('--p', '--pers', type=float, required=False, default=0.5,
                        help='Specify perturbation strength (between 0 and 1) for randomized NNI in IQTREE. '
                             'Default: 0.5')
    args = parser.parse_args()

    main(args)

import glob
import os
import shutil
from pathlib import Path
import argparse
import pandas as pd
import subprocess
import mmap
from timeit import default_timer as timer
from datetime import timedelta

__author__ = "Camilo Fuentes-Beals"
__version__ = "1.0"
__title__ = "pHMM-ASR_main"
__license__ = "GPLv3"
__author_email__ = "kmilo.f@gmail.com"


def iqtree_process(alrt, ufboot, pers, nstop, best_msa_id, best_msa_path, debug, ncores, output_path, redo):
    """

    :param alrt:
    :param ufboot:
    :param pers:
    :param nstop:
    :param best_msa_id:
    :param best_msa_path:
    :param debug:
    :param ncores:
    :param output_path:
    :param redo
    :return:
    """
    work_dir = os.path.join(output_path, 'phylo_inference')
    work_dir_asr = os.path.join(output_path, 'asr_data')
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    Path(work_dir_asr).mkdir(parents=True, exist_ok=True)

    # TODO
    # Change
    if ncores == 0:
        ncores = int(os.cpu_count() / 2)

    print('---' * 2)
    print('Phylogenetic inference directory: ', work_dir)

    iqtree_output_name = 'phylo_inf_msa_' + str(best_msa_id)
    iqtree_output_name_asr = iqtree_output_name + '_asr'
    if alrt:
        iqtree_output_name += '_alrt'
    if ufboot:
        iqtree_output_name += '_ufboot'
    if pers:
        iqtree_output_name += '_pers'
    if nstop:
        iqtree_output_name += '_nstop'

    iqtree_cmd = 'iqtree2 -s ' + best_msa_path + ' -T ' + str(ncores) + ' -st DNA'
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

    iqtree_cmd += ' -pre ' + os.path.join(work_dir, iqtree_output_name)
    print('IQTREE command: ', iqtree_cmd)
    subprocess.run(iqtree_cmd, shell=True)

    iqtree_log = os.path.join(work_dir, iqtree_output_name + '.iqtree')
    tree_file = os.path.join(work_dir, iqtree_output_name + '.treefile')

    with open(r'' + iqtree_log, 'rb', 0) as file:
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mmap_obj:
            start = mmap_obj.find(b"Best-fit model according to BIC:")
            end = mmap_obj.find(b"List")
            evo_model = str(mmap_obj[start:end - 2]).split(':')[1].strip().strip('\'')

    print('Ancestral Sequence Reconstruction')

    iqtree_asr_cmd = 'iqtree2'
    if not debug:
        iqtree_asr_cmd += ' -quiet'
    iqtree_asr_cmd += ' -s ' + best_msa_path + ' -T ' + str(ncores) + ' -asr -te ' + str(
        tree_file) + ' -m ' + evo_model + ' -pre ' + os.path.join(work_dir_asr, iqtree_output_name_asr)

    print('IQTREE ASR command: ', iqtree_asr_cmd)
    subprocess.run(iqtree_asr_cmd, shell=True)

    print('---' * 2)
    print('Analyzing ASR data.')
    asr_state_path = os.path.join(work_dir_asr, iqtree_output_name_asr + '.state')
    asr_state_df = pd.read_csv(asr_state_path, sep='\t', comment='#')
    align_length = asr_state_df['Site'].max()
    nodes = asr_state_df['Node'].unique().tolist()

    with open(os.path.join(output_path, 'asr_data.log'), 'w') as file:
        file.write('ASR information.\n')
        print('ASR information.\n')
        file.write('ASR Alignment length: ' + str(align_length) + '\n')
        print('ASR Alignment length: ' + str(align_length) + '\n')
        file.write('ASR Nodes: ' + str(len(nodes)) + '\n')
        print('ASR Nodes: ' + str(len(nodes)) + '\n')

    print('Writing ASR Fasta sequences.')
    fasta_dir = os.path.join(work_dir_asr, 'asr_fasta')
    Path(fasta_dir).mkdir(parents=True, exist_ok=True)

    asr_fasta_file = os.path.join(fasta_dir, 'msa_' + str(best_msa_id) + '_asr.fasta')
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
    print('Aligning ASR sequences.')
    aln_dir = os.path.join(fasta_dir, 'asr_align')
    Path(aln_dir).mkdir(parents=True, exist_ok=True)
    asr_fasta_msa = os.path.join(aln_dir, 'msa_' + str(best_msa_id) + '_asr.aln')

    if redo:
        print('Deleting alignment files in assigned folder.')
        res = [f.unlink() for f in Path(aln_dir).glob("*.aln") if f.is_file()]
        print(str(len(res)) + ' files deleted. ')

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


def hmm_process(output_path, asr_fasta_file, alignment_file_path, ncores, max_param, top_strand, redo, or_fasta):
    """

    :param output_path:
    :param asr_fasta_file:
    :param alignment_file_path:
    :param ncores:
    :param max_param:
    :param top_strand:
    :param redo:
    :param or_fasta:
    :return:
    """
    # Directory creation
    hmm_dir = os.path.join(output_path, 'asr_hmm_models')
    Path(hmm_dir).mkdir(parents=True, exist_ok=True)
    hmm_res_dir = os.path.join(output_path, 'asr_hmm_results')
    Path(hmm_res_dir).mkdir(parents=True, exist_ok=True)
    hmm_res_dir_or = os.path.join(hmm_res_dir, 'or_file')
    Path(hmm_res_dir_or).mkdir(parents=True, exist_ok=True)

    # Base information & HMM build - hmmbuild
    base_name = str(Path(alignment_file_path).stem)
    hmm_path = os.path.join(hmm_dir, base_name + '_profile.hmm')
    hmmbuild_cmd = 'hmmbuild --cpu ' + str(ncores) + ' -o ' + os.path.join(hmm_dir,
                                                                           base_name + '.log') + ' ' + hmm_path + ' ' + alignment_file_path
    print('hmmbuild command: ', hmmbuild_cmd)
    if os.path.exists(os.path.join(hmm_dir, base_name + '.log')):
        print('File exist - Skipping')
    else:
        subprocess.run(hmmbuild_cmd, shell=True)

    # HMM Search - nhmmer - ASR pHMM vs ASR fasta
    if ncores == 0:
        ncores = os.cpu_count()

    if redo:
        print('Deleting HMM related files in assigned folder.')
        res_log = [f.unlink() for f in Path(hmm_res_dir).glob("*.log") if f.is_file()]
        print(str(len(res_log)) + ' files deleted. ')
        res_out = [f.unlink() for f in Path(hmm_res_dir).glob("*.out") if f.is_file()]
        print(str(len(res_out)) + ' files deleted. ')

    nhmmer_cmd = 'nhmmer --cpu ' + str(ncores)
    if max_param:
        nhmmer_cmd += ' --max'
    if top_strand:
        nhmmer_cmd += ' --watson'

    nhmmer_cmd += ' -o ' + os.path.join(hmm_res_dir, base_name + '.log') + ' --tblout ' + os.path.join(
        hmm_res_dir, base_name + '.out ') + ' ' + hmm_path + ' ' + asr_fasta_file
    print('nhmmer command: ', nhmmer_cmd)
    if os.path.exists(os.path.join(hmm_res_dir, base_name + '.log')):
        print('File exist - Skipping')
    else:
        subprocess.run(nhmmer_cmd, shell=True)

    # HMM Search - nhmmer - ASR pHMm vs Original Fasta File
    if redo:
        print('Deleting HMM related files in assigned folder.')
        res_log = [f.unlink() for f in Path(hmm_res_dir_or).glob("*.log") if f.is_file()]
        print(str(len(res_log)) + ' files deleted. ')
        res_out = [f.unlink() for f in Path(hmm_res_dir_or).glob("*.out") if f.is_file()]
        print(str(len(res_out)) + ' files deleted. ')

    nhmmer_cmd_or = 'nhmmer --cpu ' + str(ncores)
    if max_param:
        nhmmer_cmd_or += ' --max'
    if top_strand:
        nhmmer_cmd_or += ' --watson'

    nhmmer_cmd_or += ' -o ' + os.path.join(hmm_res_dir_or, base_name + '.log') + ' --tblout ' + os.path.join(
        hmm_res_dir_or, base_name + '.out') + ' ' + hmm_path + ' ' + or_fasta
    print('nhmmer command: ', nhmmer_cmd_or)
    if os.path.exists(os.path.join(hmm_res_dir_or, base_name + '.log')):
        print('File exist - Skipping')
    else:
        subprocess.run(nhmmer_cmd_or, shell=True)


def prep_files(parent_folder, best_msa_id):
    # Expected files and folders paths.
    # MSA
    msa_folder = os.path.join(parent_folder, 'alignment_files')
    msa_file_name = 'train_' + str(best_msa_id) + '_sequences_pw_al_score.aln'
    msa_path = os.path.join(msa_folder, msa_file_name)
    # pHMM
    phmm_folder = os.path.join(parent_folder, 'hmm_models')
    phmm_file_name_log = 'train_' + str(best_msa_id) + '_sequences_pw_al_score.log'
    phmm_file_name_hmm = 'train_' + str(best_msa_id) + '_sequences_pw_al_score_profile.hmm'
    phmm_fname_log_path = os.path.join(phmm_folder, phmm_file_name_log)
    phmm_fname_hmm_path = os.path.join(phmm_folder, phmm_file_name_hmm)
    # Results
    phmm_results_folder = os.path.join(parent_folder, 'hmm_results')
    phmm_res_sub_folder = os.path.join(phmm_results_folder, 'test_files_nhmmer')
    phmm_results_fname_log = 'train_' + str(best_msa_id) + '_sequences_pw_al_score.log'
    phmm_results_fname_out = 'train_' + str(best_msa_id) + '_sequences_pw_al_score.out'
    phmm_res_fname_log_path = os.path.join(phmm_results_folder, phmm_results_fname_log)
    phmm_res_fname_out_path = os.path.join(phmm_results_folder, phmm_results_fname_out)
    phmm_res_sub_fname_log_path = os.path.join(phmm_res_sub_folder, phmm_results_fname_log)
    phmm_res_sub_fname_out_path = os.path.join(phmm_res_sub_folder, phmm_results_fname_out)

    print('Creating folder structure.\nMain Folder')
    top_dir = os.path.join(parent_folder, str(best_msa_id) + '_data')
    try:
        Path(top_dir).mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print('Error: ', exc)
    else:
        print('Folder created - ' + str(top_dir))

    print('Creating folder to save the best MSA file.')
    best_msa_dir = os.path.join(top_dir, 'alignment_files')
    try:
        Path(best_msa_dir).mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print('Error: ', exc)
    else:
        print('Folder created - ' + str(best_msa_dir))

    print('Creating folder to save best pHMM and results.')
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

    print('Copying best MSA to created folder.')
    best_msa_path = os.path.join(best_msa_dir, msa_file_name)
    try:
        shutil.copy2(msa_path, best_msa_path)
    except Exception as exc:
        print('Error: ', exc)
    else:
        print('File copied successfully - ')

    print('Copying pHMM related files.')
    try:
        shutil.copy2(phmm_fname_log_path, os.path.join(best_phmm_dir, phmm_file_name_log))
        shutil.copy2(phmm_fname_hmm_path, os.path.join(best_phmm_dir, phmm_file_name_hmm))

        shutil.copy2(phmm_res_fname_log_path, os.path.join(best_phmm_res_dir, phmm_results_fname_log))
        shutil.copy2(phmm_res_fname_out_path, os.path.join(best_phmm_res_dir, phmm_results_fname_out))

        shutil.copy2(phmm_res_sub_fname_log_path, os.path.join(phhm_res_dir_test, phmm_results_fname_log))
        shutil.copy2(phmm_res_sub_fname_out_path, os.path.join(phhm_res_dir_test, phmm_results_fname_out))

    except Exception as exc:
        print('Error: ', exc)
    else:
        print('Files copied successfully - ')

    with open(os.path.join(top_dir, 'best_msa_data.log'), 'w') as file:
        file.write('Best MSA ID: ' + str(best_msa_id))

    return top_dir, best_msa_path


def main(main_args):
    base_dir = os.path.normpath(main_args.b)
    n_cores = main_args.n
    redo_align = main_args.ra
    redo_search = main_args.rs

    print('--- Base Options ---')
    print('base dir: ', base_dir)
    print('CPU Cores to use:', n_cores)
    print('Redo - Alignment: ', redo_align)
    print('Redo - nhmmer: ', redo_search)
    print('debug', main_args.d)
    print('--- IQTREE Options ---')
    print('alrt: ', main_args.a)
    print('ufboot: ', main_args.ub)
    print('nstop: ', main_args.ns)
    print('pers: ', main_args.p)
    print('--- nhmmer Options ---')
    print('nhmmer max param:', main_args.m)
    print('nhmmer top strand param:', main_args.t)
    print('------\n')

    for f in glob.glob(base_dir + '/*.chk', recursive=True):
        f = Path(f)
        print('Working on :', f.stem)

        msa_data = pd.read_csv(os.path.join(f.parent, "sim_res_max_w_test_.csv"), index_col=0)
        print('\nFile loaded - ')
        start_t = timer()
        # TODO
        # change selection source
        msa_id = msa_data.sort_values(by='test_seqs', ascending=False).index[0]

        print('Id. of Best MSA: ', msa_id)
        top_dir, msa_path = prep_files(parent_folder=f.parent, best_msa_id=msa_id)

        with open(r'' + str(f), 'rb', 0) as file:
            with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mmap_obj:
                start = mmap_obj.find(b"Original fasta file: ")
                end = mmap_obj.find(b".fasta")
                or_fasta_path = str(mmap_obj[start:end]).split(' ')[3].strip('\'') + '.fasta'

        asr_fasta_file, asr_fasta_msa = iqtree_process(alrt=main_args.a, ufboot=main_args.ub, pers=main_args.p,
                                                       nstop=main_args.ns, best_msa_id=msa_id, debug=main_args.d,
                                                       ncores=n_cores, best_msa_path=msa_path, output_path=top_dir,
                                                       redo=redo_align)

        hmm_process(output_path=top_dir, alignment_file_path=asr_fasta_msa, asr_fasta_file=asr_fasta_file,
                    ncores=n_cores, redo=redo_search, top_strand=main_args.t, max_param=main_args.m,
                    or_fasta=or_fasta_path)
        end_t = timer()

        with open(os.path.join(top_dir, str(msa_id) + '.chk'), 'w') as file:
            file.write('Job done for MSA with id: ' + str(msa_id) + '\n')
            file.write('Elapsed time: ' + str(timedelta(seconds=end_t - start_t)) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline for pHMM creation and ASR.',
                                     epilog='By Camilo Fuentes-Beals. @cakofuentes')
    # Base arguments
    parser.add_argument('--b', '--basedir', required=True,
                        help='Input folder. Should be the output folder of the MSA Calculations. Add an * to search an'
                             ' entire folder with subfolders. Ex. "/path/to/main_folder/*". '
                             'Double or single quote needed. ')
    parser.add_argument('--d', '--debug', required=False, action=argparse.BooleanOptionalAction,
                        help='Whether to show or not program logs in execution.')
    # cpu arguments.
    parser.add_argument('--n', '--ncore', type=int, required=True,
                        help='Default number of CPU cores to use across the pipeline.'
                             '0 dictates to use all available cpu cores')
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
    # nhmmer options
    parser.add_argument('--m', '--max', required=False, default=False, action='store_true',
                        help='nhmmer max param. Disabled by default.')
    parser.add_argument('--t', '--ts', required=False, default=False, action='store_true',
                        help='Top strand param for nhmmer. Disabled by default')
    # redo options.
    parser.add_argument('--ra', '--redo-align', required=False, default=False, action='store_true',
                        help='Redo the alignment portion of the analysis.')
    parser.add_argument('--rs', '--redo-search', required=False, default=False, action='store_true',
                        help='Redo the nhmmer (search) portion of the analysis.')
    args = parser.parse_args()
    main(args)

import mmap

import pandas as pd
from pyfaidx import Fasta
from pathlib import Path
import glob
from tqdm import tqdm
import os
import argparse
import subprocess

"""
    Defaults
"""
gap_open_penalty = 10
gap_extension_penalty = 1


def main(main_args):
    base_dir = os.path.normpath(main_args.b)
    seq_limit = main_args.l
    print('base dir: ', base_dir)
    print('seq lim: ', seq_limit)
    for f in tqdm(glob.glob(base_dir + '/*/caljac_*.fasta', recursive=True), desc='TEs'):
        te_name = Path(f).name.split('_')[-2:][0].split('.')[0]
        te_file = Fasta(f)
        total_tes = len(te_file.keys())
        file_name = str(Path(f).name).split('.')[0]
        work_folder = str(Path(f).parent)

        mafft_mode = '--localpair'
        align_folder = os.path.join(work_folder, 'alignment_' + mafft_mode[2:])
        hmm_model = os.path.join(align_folder, 'hmm_model')
        hmm_res = os.path.join(align_folder, 'hmm_res')
        phylo_inf = os.path.join(align_folder, 'phylo_inf')
        rooted_folder = os.path.join(phylo_inf, 'rooted')
        asr_folder = os.path.join(align_folder, 'asr')
        asr_fasta = os.path.join(asr_folder, 'fasta')
        asr_aln_dir = os.path.join(asr_fasta, 'asr_align')

        asr_hmm_model = os.path.join(align_folder, 'asr_hmm_model')
        asr_hmm_res = os.path.join(align_folder, 'asr_hmm_res')

        Path(align_folder).mkdir(parents=True, exist_ok=True)
        Path(hmm_model).mkdir(parents=True, exist_ok=True)
        Path(hmm_res).mkdir(parents=True, exist_ok=True)
        Path(phylo_inf).mkdir(parents=True, exist_ok=True)
        Path(rooted_folder).mkdir(parents=True, exist_ok=True)
        Path(asr_folder).mkdir(parents=True, exist_ok=True)
        Path(asr_fasta).mkdir(parents=True, exist_ok=True)
        Path(asr_aln_dir).mkdir(parents=True, exist_ok=True)
        Path(asr_hmm_model).mkdir(parents=True, exist_ok=True)
        Path(asr_hmm_res).mkdir(parents=True, exist_ok=True)

        print('file name: ', file_name)
        print('total tes: ', total_tes)
        print('te name: ', te_name)
        print('work folder: ', work_folder)
        if total_tes < seq_limit:
            if Path(os.path.join(align_folder, file_name + '.aln')).is_file():
                print('Alignment exist, skiping')
            else:
                mafft_cmd = 'mafft ' + mafft_mode + ' --thread -1 --reorder ' + str(
                    f) + ' > ' + align_folder + '/' + file_name + '.aln'
                try:
                    subprocess.run(mafft_cmd, shell=True, check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    print(e)

            if Path(os.path.join(hmm_model, file_name + '_seqs.log')).is_file():
                print('hmm model exist, skiping')
            else:
                hmmbuild_cmd = 'hmmbuild -o ' + hmm_model + '/' + file_name + '_seqs.log ' + hmm_model + '/' + file_name + '_profile.hmm ' + align_folder + '/' + file_name + '.aln'
                try:
                    subprocess.run(hmmbuild_cmd, shell=True, check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    print(e)

            if Path(os.path.join(hmm_res, file_name + '_seqs.log')).is_file():
                print('hmm res exist, skiping')
            else:
                nhmmer_cmd = 'nhmmer --max -o ' + hmm_res + '/' + file_name + '_seqs.log --tblout ' + hmm_res + '/' + file_name + '_seqs.out ' + hmm_model + '/' + file_name + '_profile.hmm ' + str(
                    f)
                try:
                    subprocess.run(nhmmer_cmd, shell=True, check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    print(e)

            if Path(os.path.join(phylo_inf, file_name + '_phylo_inf.treefile')).is_file():
                print('phylo inf exist, skiping')
            else:
                iqtree_cmd = 'iqtree2 -s ' + align_folder + '/' + file_name + '.aln' + ' -T AUTO -st DNA -quiet -alrt 1000 -B 1000 -pre ' + str(
                    os.path.join(phylo_inf, file_name)) + '_phylo_inf'
                try:
                    subprocess.run(iqtree_cmd, shell=True, check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    print(e)

            if Path(os.path.join(rooted_folder, file_name + '_rooted.treefile')).is_file():
                print('rooted inf exist, skipping')
            else:
                tree_file = os.path.join(phylo_inf, file_name + '_phylo_inf.treefile')
                iqtree_root_cmd = 'iqtree2 -m 12.12 -s ' + align_folder + '/' + file_name + '.aln' + ' -T AUTO -quiet -te ' + str(
                    tree_file) + ' -pre ' + str(os.path.join(rooted_folder, file_name)) + '_rooted'
                try:
                    subprocess.run(iqtree_root_cmd, shell=True, check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    print(e)

            if Path(os.path.join(asr_folder, file_name + '_asr.state')).is_file():
                print('asr inf exist, skipping')
            else:
                tree_file = os.path.join(phylo_inf, file_name + '_phylo_inf.treefile')
                log = os.path.join(phylo_inf, file_name + '_phylo_inf.iqtree')
                with open(r'' + log, 'rb', 0) as file:
                    with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mmap_obj:
                        start = mmap_obj.find(b"Best-fit model according to BIC:")
                        end = mmap_obj.find(b"List")
                        evo_model = str(mmap_obj[start:end - 2]).split(':')[1].strip().strip('\'')
                print('---' * 2)
                print('\nAncestral Sequence Reconstruction')
                iqtree_asr_cmd = 'iqtree2  -s ' + align_folder + '/' + file_name + '.aln' + ' -T AUTO -asr -te ' + str(
                    tree_file) + ' -m ' + evo_model + ' -pre ' + os.path.join(asr_folder, file_name + '_asr')
                print('IQTREE ASR command: ', iqtree_asr_cmd)
                try:
                    subprocess.run(iqtree_asr_cmd, shell=True, check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    print(e)

            print('Analyzing ASR data.')
            asr_state_path = os.path.join(asr_folder, file_name + '_asr.state')
            asr_state_df = pd.read_csv(asr_state_path, sep='\t', comment='#')
            align_length = asr_state_df['Site'].max()
            nodes = asr_state_df['Node'].unique().tolist()

            asr_fasta_file = os.path.join(asr_fasta, file_name + '_asr.fasta')
            print('FASTA file will be saved in: ', asr_fasta_file)
            with open(asr_fasta_file, 'w') as asr_fasta:
                for i, node in enumerate(nodes):
                    start = align_length * i
                    end = align_length * (i + 1)
                    node_seq = ''.join(asr_state_df[start:end]['State'].values)
                    node_seq = node_seq.replace("-", "")
                    asr_fasta.write('>' + node + '\n')
                    asr_fasta.write(node_seq + '\n')
            print('---' * 2)
            print('Aligning ASR sequences.')
            asr_fasta_msa = os.path.join(asr_aln_dir, file_name + '_asr.aln')
            mafft_cmd = 'mafft --thread -1 --reorder ' + asr_fasta_file + ' > ' + asr_fasta_msa
            print('MAFFT command: ', mafft_cmd)
            if os.path.exists(asr_fasta_msa):
                if os.stat(asr_fasta_msa).st_size > 0:
                    print('File exist. Skipping.')
                else:
                    subprocess.run(mafft_cmd, shell=True)
            else:
                subprocess.run(mafft_cmd, shell=True)

            if Path(os.path.join(asr_hmm_model, file_name + '_asr.log')).is_file():
                print('asr hmm model exist, skiping')
            else:
                asr_hmmbuild_cmd = 'hmmbuild -o ' + asr_hmm_model + '/' + file_name + '_asr.log ' + asr_hmm_model + '/' + file_name + '_asr_profile.hmm ' + asr_aln_dir + '/' + file_name + '_asr.aln'
                try:
                    subprocess.run(asr_hmmbuild_cmd, shell=True, check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    print(e)

            if Path(os.path.join(asr_hmm_res, file_name + '_asr.log')).is_file():
                print('asr hmm res exist, skiping')
            else:
                asr_nhmmer_cmd = 'nhmmer --max -o ' + asr_hmm_res + '/' + file_name + '_asr.log --tblout ' + asr_hmm_res + '/' + file_name + '_asr.out ' + asr_hmm_model + '/' + file_name + '_asr_profile.hmm ' + str(
                    f)
                try:
                    subprocess.run(asr_nhmmer_cmd, shell=True, check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    print(e)

        print('----')


if __name__ == '__main__':
    desc = 'alignment of multiple TE sequences.'
    parser = argparse.ArgumentParser(description=desc, epilog='By Camilo Fuentes-Beals. @cakofuentes')
    # Base arguments
    parser.add_argument('--b', '--basedir', required=True, help='Input folder.')
    parser.add_argument('--l', '--limit', type=int, default=1000, required=False,
                        help='Upper limit of number of sequences. Default 1000.')

    args = parser.parse_args()

    main(args)

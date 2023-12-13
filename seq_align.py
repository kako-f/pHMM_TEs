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
        Path(align_folder).mkdir(parents=True, exist_ok=True)

        hmm_model = os.path.join(align_folder, 'hmm_model')
        hmm_res = os.path.join(align_folder, 'hmm_res')
        Path(align_folder).mkdir(parents=True, exist_ok=True)
        Path(hmm_model).mkdir(parents=True, exist_ok=True)
        Path(hmm_res).mkdir(parents=True, exist_ok=True)

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

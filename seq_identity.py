import pandas as pd
from pyfaidx import Fasta
from pathlib import Path
import glob
from tqdm.notebook import tqdm
import parasail
import os
import numpy as np
import argparse

"""
    Defaults
"""
gap_open_penalty = 10
gap_extension_penalty = 1


def main(main_args):
    base_dir = os.path.normpath(main_args.b)
    print('base dir', base_dir)
    for f in tqdm(glob.glob(base_dir + '/*/caljac_*.fasta', recursive=True), desc='TEs'):
        te_file = Fasta(f)
        identity_perc_dict = []
        seq_gaps_dict = []
        work_folder = str(Path(f).parent)
        if Path(os.path.join(work_folder, 'identity_percentage.csv')).exists():
            print('File exist, skipping')
        else:
            for seq in tqdm(te_file.values(), desc='Pairwise align'):
                for seq2 in te_file.values():
                    result = parasail.sw_trace_scan_sat(str(seq), str(seq2), gap_open_penalty, gap_extension_penalty,
                                                        parasail.dnafull)
                    identity = (result.traceback.comp.count('|') * 100) / len(result.traceback.ref)
                    gaps = (result.traceback.comp.count(' ') * 100) / len(result.traceback.ref)
                    identity_perc_dict.append(identity)
                    seq_gaps_dict.append(gaps)
            identity_perc_matrix = np.reshape(identity_perc_dict, (len(te_file.values()), len(te_file.values())))
            seq_gaps_matrix = np.reshape(seq_gaps_dict, (len(te_file.values()), len(te_file.values())))
            identity_perc_df = pd.DataFrame(identity_perc_matrix, )
            seq_gaps_df = pd.DataFrame(seq_gaps_matrix, )
            identity_data = pd.DataFrame()
            gaps_data = pd.DataFrame()
            identity_data = identity_data.assign(
                identity_perc_median=identity_perc_df.median(numeric_only=True, axis=1))
            gaps_data = gaps_data.assign(seq_gaps_median=seq_gaps_df.median(numeric_only=True, axis=1))

            identity_data = identity_data.assign(identity_perc_variance=identity_perc_df.var(numeric_only=True, axis=1))
            gaps_data = gaps_data.assign(seq_gaps_variance=seq_gaps_df.var(numeric_only=True, axis=1))

            identity_data = identity_data.assign(identity_perc_std=identity_perc_df.std(numeric_only=True, axis=1))
            gaps_data = gaps_data.assign(seq_gaps_std=seq_gaps_df.std(numeric_only=True, axis=1))

            identity_data.to_csv(os.path.join(work_folder, 'identity_percentage.csv'), )
            gaps_data.to_csv(os.path.join(work_folder, 'gap_percentage.csv'), )


if __name__ == '__main__':
    desc = 'Pairwise alignment of multiple sequences in fasta files.'
    parser = argparse.ArgumentParser(description=desc, epilog='By Camilo Fuentes-Beals. @cakofuentes')
    # Base arguments
    parser.add_argument('--b', '--basedir', required=True, help='Input folder.')

import pandas as pd
from pathlib import Path
import subprocess
from Bio import AlignIO
import sys, argparse, os

import panel as pn
pn.extension()

module_path = str(Path.cwd() / "modules")
if module_path not in sys.path:
    sys.path.append(module_path)
from modules import functions


def main(main_args):
    print(main_args)
    print('Sequence length: ', main_args.l)
    print('Start: ', main_args.sa)
    print('Step: ', main_args.st)
    print('Changes Step: ', main_args.cst)
    print('Repetitions: ', main_args.nr)
    print('Sequence by repetition: ', main_args.ns)
    print('Range: ', main_args.range)
    print('Output folder: ', main_args.o)

    main_folder = Path(main_args.o)
    Path(main_folder).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(main_folder, 'sim_data.txt'), 'w') as f:
        f.write('Sequence length: ' + str(main_args.l))
        f.write('Start: ' + str(main_args.sa))
        f.write('Step: ' + str(main_args.st))
        f.write('Changes Step: ' + str(main_args.cst))
        f.write('Repetitions: ' + str(main_args.nr))
        f.write('Sequence by repetition: ' + str(main_args.ns))
        f.write('Range: ' + str(main_args.range))
        f.write('Output folder: ' + str(main_args.o))

    if main_args.range:
        seqlen = [*range(int(main_args.sa), int(main_args.l) + int(main_args.st), int(main_args.st))]
        print(seqlen)
    else:
        seqlen = [int(main_args.l)]
        print(seqlen)

    for _len in seqlen:
        baseseq = functions.random_dna_seq(int(_len), )[0]
        nchanges = [*range(0, _len + int(main_args.cst), int(main_args.cst))]
        for ch in nchanges:
            print(ch)
            seq_folder = Path(
                str(main_folder) + '/seqs_rep_' + str(main_args.nr) + '_len_' + str(_len) + '_changes_' + str(ch))
            Path(seq_folder).mkdir(parents=True, exist_ok=True)

            hmm_folder = Path(
                str(main_folder) + '/hmm_rep_' + str(main_args.nr) + '_len_' + str(_len) + '_changes_' + str(ch))
            Path(hmm_folder).mkdir(parents=True, exist_ok=True)

            hmmres_folder = Path(str(hmm_folder) + '/res')
            Path(hmmres_folder).mkdir(parents=True, exist_ok=True)
            prep_sequences = functions.fragmentation(base_seq=baseseq, rep=int(main_args.nr),
                                                     total_seq=int(main_args.ns), changes=ch)
            for i, seqlist in enumerate(prep_sequences):
                with open(str(seq_folder) + '/set_' + str(i) + '_seqs_' + str(main_args.ns) + '_len_' + str(
                        _len) + '_changes_' + str(ch) + '.fasta', 'w') as file:
                    file.write('>base_seq\n' + baseseq + '\n')
                    for x, seq in enumerate(seqlist):
                        file.write('>seq_' + str(x) + '\n')
                        file.write(str(seq) + '\n')
            for path in seq_folder.glob("**/*.fasta"):
                mafft_cmd = 'mafft --quiet --thread -1 --auto ' + str(path) + ' > ' + str(path).replace('fasta', 'aln')
                subprocess.run(mafft_cmd, shell=True)
            for path in seq_folder.glob("**/*.fasta"):
                fasta_file_name = path.stem
                hmmbuild_cmd = 'hmmbuild -o ' + str(hmm_folder) + '/' + fasta_file_name + '.log ' + str(
                    hmm_folder) + '/' + fasta_file_name + '_profile.hmm ' + str(path).replace('fasta', 'aln')
                subprocess.run(hmmbuild_cmd, shell=True)
            for path in seq_folder.glob("**/*.fasta"):
                nhmmer_cmd = 'nhmmer --cpu 8 --watson -o ' + str(hmmres_folder) + '/' + str(
                    path.stem) + '.log --tblout ' + str(hmmres_folder) + '/' + str(path.stem) + '_res.out ' + str(
                    hmm_folder) + '/' + str(path.stem) + '_profile.hmm ' + str(path)
                subprocess.run(nhmmer_cmd, shell=True)
    return main_folder, main_args.l, main_args.cst


def process(data_main_folder, seqlen, changes):
    simdata = []
    with open(os.path.join(data_main_folder, 'sim_data.txt')) as file:
        for line in file.readlines():
            simdata.append(line.strip())
    seq_len = simdata[0].split(':')[1].strip()
    rep_ = simdata[4].split(':')[1].strip()
    s_by_rep_ = simdata[5].split(':')[1].strip()
    out_folder = simdata[7].split(':')[1].strip()

    data_dict = {
        'seqlen': [],
        'changes': [],
        'aln_len': [],
        'query_len': [],
        'nseqs_target': [],
        'nhits_hmm': [],
        'e_seq': []
    }
    for path in data_main_folder.glob("**/res/*.log"):
        cre_seq_len = str(path.stem).split('_')[5]
        seq_changes = str(path.stem).split('_')[7]
        aln_file = out_folder + '/seqs_rep_' + rep_ + '_len_' + cre_seq_len + '_changes_' + seq_changes + '/' + str(
            path.stem) + '.aln'
        align = AlignIO.read(aln_file, "fasta")
        with open(path) as f:
            lines = f.readlines()
            query_data = [l for l in lines if 'Query model' in l]
            target_data = [l for l in lines if 'Target sequences' in l]
            hit_dat = [l for l in lines if 'Total number of hits' in l]
            hmm_query_data = query_data[0].split(':')[1].strip().split('(')[1].split(' ')[0].strip()
            hmm_target_data = target_data[0].split(':')[1].strip().split('(')[0].strip()
            hmm_hit_data = hit_dat[0].split(':')[1].strip().split('(')[0].strip()
            data_dict['seqlen'].append(int(cre_seq_len))
            data_dict['changes'].append(int(seq_changes))
            data_dict['aln_len'].append(align.get_alignment_length())
            data_dict['query_len'].append(int(hmm_query_data))
            data_dict['nseqs_target'].append(int(hmm_target_data))
            data_dict['nhits_hmm'].append(int(hmm_hit_data))
            data_dict['e_seq'].append(int(str(path.stem).split('_')[3]) + 1)
    df_data = pd.DataFrame.from_dict(data_dict)
    df_data.to_csv(os.path.join(out_folder,
                                'all_sim_data_seqlen_' + seq_len + '_rep_' + rep_ + 'seq_by_rep_' + s_by_rep_ + '.csv'))
    condensed_data = {
        'changes': [],
        'nhits_mean': [],
        'std': [],
        'min': [],
        'max': [],
        'seqlen': [],
        'exp_seq': []
    }
    for x in df_data.groupby('seqlen'):
        for y in x[1].groupby('changes'):
            condensed_data['changes'].append(y[1]['changes'].unique().tolist()[0])
            condensed_data['nhits_mean'].append(y[1]['nhits_hmm'].mean())
            condensed_data['std'].append(y[1]['nhits_hmm'].std())
            condensed_data['min'].append(y[1]['nhits_hmm'].min())
            condensed_data['max'].append(y[1]['nhits_hmm'].max())
            condensed_data['seqlen'].append(y[1]['seqlen'].unique()[0])
            condensed_data['exp_seq'].append(y[1]['e_seq'].unique()[0])
    con_df = pd.DataFrame.from_dict(condensed_data)
    con_df = con_df.assign(high=con_df['nhits_mean'] + con_df['std'])
    con_df = con_df.assign(low=con_df['nhits_mean'] - con_df['std'])
    con_df.to_csv(os.path.join(out_folder,
                               'condensed_sim_data_seqlen_' + seq_len + '_rep_' + rep_ + 'seq_by_rep_' + s_by_rep_ + '.csv'))

    # dir_data = {
    #     'path': [],
    #     'len': [],
    #     'nchanges': []
    # }

    # for path in Path(out_folder).glob("**/*.fasta"):
    #     dir_data['path'].append(str(path))
    #     dir_data['len'].append(int(str(path.stem).split('_')[5]))
    #     dir_data['nchanges'].append(int(str(path.stem).split('_')[7]))
    #
    # dir_data_df = pd.DataFrame.from_dict(dir_data)
    # for x in dir_data_df.groupby('len'):
    #
    #     for y in x[1].groupby('nchanges'):
    #         print(y[1]['len'].iloc[0])
    #         print(y[1]['nchanges'].iloc[0])
    #         fasta_len = {
    #             'len': []
    #         }
    #         for p in y[1]['path']:
    #             fasta_file = Fasta(p)
    #             for key in fasta_file.keys():
    #                 fasta_len['len'].append(len(str(fasta_file[key])))
    #         print(str(Path(p).stem))
    #         fdf = pd.DataFrame.from_dict(fasta_len)
    #         xx = fdf.hvplot.hist(title='Length: ' + str(y[0]) + ' Changes: ' + str(y[0]))
    #         hvplot.save(xx, os.path.join(out_folder, str(Path(p).stem).replace('set', 'rep') + '_plot.html'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--r', '--range', required=True, default=True, action=argparse.BooleanOptionalAction)
    # parser.add_argument('--r', dest='range', action='store_true')
    # parser.add_argument('--no-r', dest='range', action='store_false')
    parser.set_defaults(range=True)
    parser.add_argument('--sa', '--start', required=False, default=100)
    parser.add_argument('--st', '--step', required=False, default=100)
    parser.add_argument('--cst', '--ch_step', required=False, default=10)
    parser.add_argument('--l', '--seqlen', required=True)
    parser.add_argument('--nr', '--nrep', required=True)
    parser.add_argument('--ns', '--nseq', required=True)
    parser.add_argument('--o', '--output', required=True)

    args = parser.parse_args()

    mf, seq_len, ch = main(args)
    process(mf, seq_len, ch)

import numpy as np
import random


def random_dna_seq(length, p_gc=None, p_all=None, exact=False):
    """
    Creating random dna sequences.

    :param length: base length of the sequence.
    :param p_gc: GC percentage. As Float.
    :param p_all: percentage of all the nucleotides - order: ACGT
    :exact: When a nt % is given, the sequence is created by shuffling instead of random choice.
    :return:
    """

    if p_all:
        rng = np.random.default_rng()
        if exact:
            s = list(
                'A' * int(p_all[0] * length) + 'C' * int(p_all[1] * length) + 'G' * int(p_all[2] * length) + 'T' * int(
                    p_all[3] * length))
            rng.shuffle(s)
            res_dna_seq = ''.join(s)
            nt_p = p_all
        else:
            p = np.array(p_all)
            p /= p.sum()
            nt_p = p
            dna_seq = rng.choice(['A', 'C', 'G', 'T'], size=length, p=nt_p)
            res_dna_seq = ''.join(dna_seq)
    elif p_gc:
        rng = np.random.default_rng()
        p_at = 1.0 - p_gc
        nt_p = [(p_at / 2), (p_gc / 2), (p_gc - (p_gc / 2)), (p_at - (p_at / 2))]
        if exact:
            s = list(
                'A' * int(nt_p[0] * length) + 'C' * int(nt_p[1] * length) + 'G' * int(nt_p[2] * length) + 'T' * int(
                    nt_p[3] * length))
            rng.shuffle(s)
            res_dna_seq = ''.join(s)
        else:
            dna_seq = rng.choice(['A', 'C', 'G', 'T'], size=length, p=nt_p)
            res_dna_seq = ''.join(dna_seq)
    if p_gc == None and p_all == None:
        nt_p_rng = np.random.default_rng()
        nt_p = nt_p_rng.random((4,))
        nt_p = nt_p / nt_p.sum()
        dna_seq = nt_p_rng.choice(['A', 'C', 'G', 'T'], size=length, p=nt_p)
        res_dna_seq = ''.join(dna_seq)

    return res_dna_seq, nt_p


def insert_str(string, str_to_insert, index):
    return string[:index] + str_to_insert + string[index:]


def replace_str_index(text, index=0, replacement=''):
    return f'{text[:index]}{replacement}{text[index + 1:]}'


def get_base_frequencies(dna):
    """
		Return frequencies of the dna bases.
	"""

    return {base: dna.count(base) / float(len(dna)) for base in 'ACGT'}


def random_mutation(type, selected_seqs, rep, total_seq, n_changes, size=1, min_seq_size=10, te_mimic=False,
                    predefined_len=None, te_nt_perc=None, base_seq=None, ):
    """
        :type: Mutation type, either: insertion, deletion or indel
        :base_seq: Base sequence to be used as a starting point
        :selected_seqs: Number of selected sequences to be used. <= total_seqs
        :rep: Total: repetitions of modifications to be made
        :total_seq: Amount of sequences to be created in each repetition
        :n_changes: amount of changes according to type
        :size: Wether to do a point mutation (1nt) or greater.
        :min_seq_size: Minimun length after deletions.
        :predefined_len: When mimicking a TE, this is the set of length extracted from a FASTA file. Could be a sample or the entire set.
        :te_nt_perc: median of ACGT percentages extracted from the TE fasta file.

        When mimicking a TE, the variables predefined_len and te_nt_prec are used in the random_dna_seq function to create a set of sequences with those values.

        :return: List of created sequences
    """
    nt = ['A', 'C', 'G', 'T']
    base_full_list = []
    if total_seq == selected_seqs:
        selected_seqs = list(range(0, total_seq))
    if te_mimic:
        if predefined_len == None and te_nt_perc == None:
            raise Exception("Sorry, predefined_len and te_nt_perc must be defined.")
        else:
            for _ in range(rep):
                seq_list = []
                for p_len in predefined_len:
                    seq_list.append(random_dna_seq(length=p_len,
                                                   p_all=[te_nt_perc['A'], te_nt_perc['C'], te_nt_perc['G'],
                                                          te_nt_perc['T']])[0])
                base_full_list.append(seq_list)
    elif base_seq == None:
        raise Exception("Sorry, base_seq must be defined")
    else:
        for _ in range(rep):
            seq_list = []
            seq_list = [base_seq] * total_seq
            base_full_list.append(seq_list)

        if type == 'insertion':
            for seq_set in base_full_list:
                for n_seq in selected_seqs:
                    positions = random.choices(range(0, len(seq_set[n_seq])), k=n_changes, )
                    new_string = seq_set[n_seq]
                    for y in positions:
                        ins = random.choices(nt, k=size)
                        repl = ''.join(ins)
                        new_string = insert_str(string=new_string, str_to_insert=repl, index=y)
                    seq_set[n_seq] = new_string
        elif type == 'deletion':
            for seq_set in base_full_list:
                for n_seq in selected_seqs:
                    positions = random.choices(range(0, len(seq_set[n_seq])), k=n_changes, )
                    new_string = seq_set[n_seq]
                    for y in positions:
                        if len(new_string) <= min_seq_size:
                            pass
                        else:
                            new_string = replace_str_index(new_string, y, )
                    seq_set[n_seq] = new_string
        elif type == 'mutation':
            for seq_set in base_full_list:
                for n_seq in selected_seqs:
                    positions = random.choices(range(0, len(seq_set[n_seq])), k=n_changes, )
                    new_string = seq_set[n_seq]
                    for y in positions:
                        repl = random.choices(nt, k=size)[0]
                        new_string = replace_str_index(new_string, y, replacement=repl)
                    seq_set[n_seq] = new_string
        elif type == 'indels':
            for seq_set in base_full_list:
                for n_seq in selected_seqs:
                    positions = random.choices(range(0, len(seq_set[n_seq])), k=n_changes, )
                    new_string = seq_set[n_seq]
                    for y in positions:
                        if random.random() < 0.5:
                            ins = random.choices(nt, k=size)
                            repl = ''.join(ins)
                            new_string = insert_str(string=new_string, str_to_insert=repl, index=y)
                            seq_set[n_seq] = new_string
                        else:
                            if len(new_string) <= min_seq_size:
                                pass
                            else:
                                new_string = replace_str_index(new_string, y, )
                            seq_set[n_seq] = new_string
        elif type == 'none':
            return base_full_list
    return base_full_list
import numpy as np


def random_dna_seq(length, p_gc=None, r_length=False):
    """

    :param length:
    :param p_gc:
    :param r_length:
    :return:
    """
    initial_length = length
    res_dna_seq = ''
    nt_p = ''
    if r_length:
        len_rng = np.random.default_rng()
        new_length = len_rng.integers(low=0, high=initial_length, endpoint=True)
        length = new_length
    else:
        pass

    # p_gc float, referring to % GC
    # % AT will be subtracted accordingly
    # - 
    # If None, a list with random probabilities will be created 
    # list of percentages = [A,C,G,T]

    if p_gc:
        rng = np.random.default_rng()
        p_at = 1.0 - p_gc
        nt_p = [(p_at / 2), (p_gc / 2), (p_gc - (p_gc / 2)), (p_at - (p_at / 2))]
        dna_seq = rng.choice(['A', 'C', 'G', 'T'], size=length, p=nt_p)
        res_dna_seq = ''.join(dna_seq)
    elif p_gc is None:
        nt_p_rng = np.random.default_rng()
        nt_p = nt_p_rng.random((4,))
        nt_p = nt_p / nt_p.sum()
        dna_seq = nt_p_rng.choice(['A', 'C', 'G', 'T'], size=length, p=nt_p)
        res_dna_seq = ''.join(dna_seq)

    return res_dna_seq, nt_p


def get_base_frequencies(dna):
    """
    :param dna:
    :return:
    """
    return {base: dna.count(base) / float(len(dna)) for base in 'ACGT'}


def fragmentation_with_distr(base_seq, rep, total_seq, dist_data):
    """
        Creating random sequences from a distribution.
        Distribution must be based on data and be a sci-py object.

        :base_seq: Base sequence to be used as a starting point
        :rep: Total: repetitions of modifications to be made
        :total_seq: Amount of sequences to be created in each repetition.
        :dist_data: Distribution of data to be used as template for length values.

        :return: List of created sequences
    """

    rng = np.random.default_rng()

    full_list = []
    for _ in range(rep):
        seq_list = []
        for i in range(total_seq):
            x = list(base_seq)
            len_data = dist_data.rvs(size=1)
            frag_size = int(abs(len_data[0]))
            while frag_size > len(base_seq):
                len_data = dist_data.rvs(size=1)
                frag_size = int(abs(len_data[0]))
            positions = rng.choice(len(base_seq), len(base_seq) - frag_size, replace=False)
            y = np.delete(x, positions)
            seq_list.append(''.join(y))
        full_list.append(seq_list)
    return full_list

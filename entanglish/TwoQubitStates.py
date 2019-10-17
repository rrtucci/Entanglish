import numpy as np
from entanglish.DenMat import *


class TwoQubitStates:
    """
    This class has no constructor. It contains only static methods. Its
    methods deal with entanglement of 2 qubit states.

    """
    @staticmethod
    def bell_key_set():
        """
        This method returns a set of 4 strings which label the 4 Bell basis
        states.

        Returns
        -------
        set[str]

        """
        return {'==+', '==-', '!=+', '!=-'}

    @staticmethod
    def get_bell_basis_st_vec(key):
        """
        This method returns one Bell basis state out of 4 possible ones. The
        4 Bell basis states are orthonormal maximally entangled 2 qubit
        states.

        |==+> = 1/sqrt(2)[|00> + |11>]
        |==-> = 1/sqrt(2)[|00> - |11>]
        |!=+> = 1/sqrt(2)[|01> + |10>]
        |!=-> = 1/sqrt(2)[|01> - |10>]

        Parameters
        ----------
        key : str
            either '==+', '==-', '!=+', or '!=-'

        Returns
        -------
        np.ndarray
            shape = (4,)

        """
        assert key in TwoQubitStates.bell_key_set()
        # row_shape = (2, 2)
        st_vec1 = np.zeros((4,), dtype=complex)
        st_vec2 = np.zeros((4,), dtype=complex)
        if key[0] == '=':
            st_vec1[0] = 1  # |0>|0>
            st_vec2[3] = 1  # |1>|1>
        else:
            st_vec1[1] = 1  # |0>|1>
            st_vec2[2] = 1  # |1>|0>
        if key[-1] == '+':
            sign1 = +1
        else:
            sign1 = -1

        return (st_vec1 + sign1 * st_vec2) / np.sqrt(2)

    @staticmethod
    def get_bell_mixture(prob_dict):
        """
        This method returns a DenMat which is constructed as a linear
        combination, with coefficients prob_dict, of the Bell basis states
        written as density matrices.

        Parameters
        ----------
        prob_dict : dict[str, float]

        Returns
        -------
        DenMat

        """
        ut.assert_is_prob_dist(np.array(list(prob_dict.values())))
        assert set(prob_dict.keys()) == TwoQubitStates.bell_key_set()
        dm = DenMat(4, (2, 2))
        dm.set_arr_to_zero()
        for key in prob_dict.keys():
            st_vec = TwoQubitStates.get_bell_basis_st_vec(key)
            dm.arr += np.outer(st_vec, np.conj(st_vec)) * prob_dict[key]
        return dm

    @staticmethod
    def get_prob_dict_of_werner_mixture(fid):
        """
        This method returns prob_dict where prob_dict["==+"]=fid=fidelity
        and all other values of prob_dict are equal to (1 - fid) / 3.

        Parameters
        ----------
        fid : float

        Returns
        -------
        dict[str, float]

        """
        assert 0 <= fid <= 1
        prob_dict = {}
        for key in TwoQubitStates.bell_key_set():
            if key == '==+':
                prob_dict[key] = fid
            else:
                prob_dict[key] = (1 - fid) / 3
        return prob_dict

    @staticmethod
    def get_known_formation_entang_of_bell_mixture(prob_dict):
        """
        This method returns the known value (according to Ref.1) for the
        entanglement of formation for a Bell mixture with coefficients
        prob_dict.

        References
        ----------

        1. C.H. Bennett, D.P. DiVincenzo, J. Smolin, W.K. Wootters, “Mixed
        state entanglement and quantum error correction”,
        https://arxiv.org/abs/quant-ph/9604024

        Parameters
        ----------
        prob_dict : dict[str, float]

        Returns
        -------
        float

        """
        assert set(prob_dict.keys()) == TwoQubitStates.bell_key_set()
        p_max = max(prob_dict.values())
        if p_max < 1/2:
            t = 0
        else:
            t = (2*p_max - 1)**2
        u = (1+np.sqrt(1-t))/2
        return ut.get_entropy_from_probs(np.array([u, 1-u]))


if __name__ == "__main__":
    from entanglish.EntangCase import *
    from entanglish.PureStEnt import *
    from entanglish.SquashedEnt import *

    def main():
        print('4 Bell Basis states**********************')
        for key in TwoQubitStates.bell_key_set():
            st_vec = TwoQubitStates.get_bell_basis_st_vec(key)
            dm = DenMat(4, (2, 2))
            dm.set_arr_from_st_vec(st_vec)
            ecase = PureStEnt(dm)
            pf = ecase.get_entang_profile()
            print('----------key:', key)
            print("st_vec=\n", st_vec)
            ecase.print_entang_profiles([pf], dm.row_shape)
        print("werner mixture*******************************")
        prob_dict = TwoQubitStates.get_prob_dict_of_werner_mixture(.7)
        print("prob_dict=", prob_dict)
        dm = TwoQubitStates.get_bell_mixture(prob_dict)
        print('arr=\n', dm.arr)
        print("formation_entang=",
              TwoQubitStates.get_known_formation_entang_of_bell_mixture(
                  prob_dict))
        num_hidden_states = 4
        num_ab_steps = 5
        print('num_hidden_states=', num_hidden_states)
        print('num_ab_steps=', num_ab_steps)
        ecase = SquashedEnt(dm, num_hidden_states, num_ab_steps,
            recursion_init="equi-diag", verbose=True)
        print('squashed_entang_0=', ecase.get_entang({0}))


    main()

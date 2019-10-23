import numpy as np
from entanglish.DenMat import *


class TwoQubitState:
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
        4 Bell basis states are orthonormal, maximally entangled, 2 qubit
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
        assert key in TwoQubitState.bell_key_set()
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
    def get_bell_basis_diag_dm(fid, prob_dict=None):
        """
        This method returns a DenMat which is constructed as a linear
        combination, with coefficients prob_dict, of the Bell basis state
        projection operators. So the den matrix returned is diagonal in the
        Bell basis.

        If prob_dict is not None, use it and ignore value of fid. If
        prob_dict is None, use prob_dict for an "isotropic" Werner state
        with fidelity fid. That is, prob_dict[ "==+"]=fid, prob_dict[x]=(
        1-fid)/3 for all x other than '==+"

        Parameters
        ----------
        fid : float
            fidelity.
        prob_dict : dict[str, float]|None

        Returns
        -------
        DenMat

        """
        if prob_dict:
            assert ut.is_prob_dist(np.array(list(prob_dict.values())))
            assert set(prob_dict.keys()) == TwoQubitState.bell_key_set()
        else:
            assert 0 <= fid <= 1
            prob_dict = {}
            for key in TwoQubitState.bell_key_set():
                if key == '==+':
                    prob_dict[key] = fid
                else:
                    prob_dict[key] = (1 - fid) / 3

        dm = DenMat(4, (2, 2))
        dm.set_arr_to_zero()
        for key in prob_dict.keys():
            st_vec = TwoQubitState.get_bell_basis_st_vec(key)
            dm.arr += np.outer(st_vec, np.conj(st_vec)) * prob_dict[key]
        return dm

    @staticmethod
    def get_time_reversed_dm(dm):
        """
        This method returns a DenMat which is the time reversed version of a
        DenMat dm for 2 qubits.

        Parameters
        ----------
        dm : DenMat

        Returns
        -------
        DenMat

        """
        assert dm.num_rows == 4
        sigy = np.array([[0, -1j], [1j, 0]])
        dm_sigyy = DenMat(4, (2, 2), arr=ut.kron_prod([sigy, sigy]))
        return dm_sigyy*dm.conj()*dm_sigyy

    @staticmethod
    def get_concurrence(dm):
        """
        This method returns the concurrence of a DenMat dm for 2 qubits.

        Parameters
        ----------
        dm : DenMat

        Returns
        -------
        float

        """
        assert dm.num_rows == 4
        dm_trev = TwoQubitState.get_time_reversed_dm(dm)
        dm_root = dm.sqrt()
        dm2 = dm_root*dm_trev*dm_root
        evas = np.sqrt(np.linalg.eigvalsh(dm2.arr))
        max_pos = np.argmax(evas)
        x = 0
        for pos in range(4):
            if pos == max_pos:
                x += evas[pos]
            else:
                x -= evas[pos]
        return max(0, x)

    @staticmethod
    def get_known_formation_entang(dm):
        """
        This method returns the known formation entanglement for a DenMat dm
        for 2 qubits.

        Parameters
        ----------
        dm : DenMat

        Returns
        -------
        float

        """
        assert dm.num_rows == 4
        conc = TwoQubitState.get_concurrence(dm)
        u = (1+np.sqrt(1-conc**2))/2
        return ut.get_entropy_from_probs(np.array([u, 1-u]))


if __name__ == "__main__":
    from entanglish.EntangCase import *
    from entanglish.PureStEnt import *
    from entanglish.SquashedEnt import *

    def main():
        print('4 Bell Basis states**********************')
        for key in TwoQubitState.bell_key_set():
            st_vec = TwoQubitState.get_bell_basis_st_vec(key)
            dm = DenMat(4, (2, 2))
            dm.set_arr_from_st_vec(st_vec)
            ecase = PureStEnt(dm)
            pf = ecase.get_entang_profile()
            print('----------key:', key)
            print("st_vec=\n", st_vec)
            ecase.print_entang_profiles([pf], dm.row_shape)
        print("*******************************")
        dm1 = TwoQubitState.get_bell_basis_diag_dm(fid=.7)
        # print('arr=\n', dm1.arr)
        np.random.seed(123)
        dm2 = DenMat(4, (2, 2))
        dm2.set_arr_to_rand_den_mat(np.array([.1, .2, .3, .4]))
        dm3 = DenMat(4, (2, 2))
        dm3.set_arr_to_rand_den_mat(np.array([.1, .1, .1, .7]))
        for dm in [dm1, dm2, dm3]:
            print("----------new dm")
            print("formation_entang=",
                  TwoQubitState.get_known_formation_entang(dm))

    main()

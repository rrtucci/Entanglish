import numpy as np
from entanglish.DenMat import *


class OtherStates:
    """
    This class has no constructor. It contains only static methods. Its
    methods return various quantum states which we decided were too simple
    to merit their own individual class.

    """

    @staticmethod
    def get_bell_basis_st_vec(bits_are_equals, mid_sign):
        """
        This method returns one Bell Basis state out of 4 possible ones. The
        4 Bell Basis states are maximally entangled 2 qubit states.

        Parameters
        ----------
        bits_are_equals : bool
        mid_sign : str
            either '+' or '-'

        Returns
        -------
        np.ndarray
            shape = (4,)

        """
        row_shape = (2, 2)
        st_vec1 = np.zeros((4,), dtype=complex)
        st_vec2 = np.zeros((4,), dtype=complex)
        if bits_are_equals:
            st_vec1[0] = 1  # |0>|0>
            st_vec2[3] = 1  # |1>|1>
        else:
            st_vec1[1] = 1  # |0>|1>
            st_vec2[2] = 1  # |1>|0>
        if mid_sign == '+':
            sign1 = +1
        elif mid_sign == '-':
            sign1 = -1
        else:
            assert False

        return (st_vec1 + sign1*st_vec2)/np.sqrt(2)

    @staticmethod
    def get_den_mat_with_bound_entang(p):
        """
        This method returns a DenMat with num_rows = 8 and row_shape = (2,
        4) that is known to have bound entanglement.

        Parameters
        ----------
        p : float
         a probability, 0 < p < 1

        Returns
        -------
        DenMat

        """
        num_rows = 8
        row_shape = (2, 4)
        a = (1 + p)/2
        b = np.sqrt(1 - p**2)/2
        arr1 = np.array([[p, 0, 0, 0, 0, p, 0, 0],
                         [0, p, 0, 0, 0, 0, p, 0],
                         [0, 0, p, 0, 0, 0, 0, p],
                         [0, 0, 0, p, 0, 0, 0, 0],
                         [0, 0, 0, 0, a, 0, 0, b],
                         [p, 0, 0, 0, 0, p, 0, 0],
                         [0, p, 0, 0, 0, 0, p, 0],
                         [0, 0, p, 0, b, 0, 0, a]])
        return DenMat(num_rows, row_shape, arr1/np.trace(arr1))


if __name__ == "__main__":
    from entanglish.EntangCase import *
    from entanglish.PureStEnt import *
    from entanglish.SquashedEnt import *

    def main():
        print('4 Bell Basis states**********************')
        for bits_are_equal in [True, False]:
            for mid_sign in ['+', '-']:
                st_vec = OtherStates.get_bell_basis_st_vec(bits_are_equal, mid_sign)
                dm = DenMat(4, (2, 2))
                dm.set_arr_from_st_vec(st_vec)
                ecase = PureStEnt(dm)
                pf = ecase.get_entang_profile()
                print('----------bits_are_equal=', bits_are_equal,
                      ', mid_sign=', mid_sign)
                print("st_vec=\n", st_vec)
                ecase.print_entang_profiles([pf], dm.row_shape)
        print('bound entang state **********************')
        dm_bd = OtherStates.get_den_mat_with_bound_entang(.5)
        num_hidden_states = 5
        num_ab_steps = 30
        ecase = SquashedEnt(dm_bd, num_hidden_states, num_ab_steps,
                            verbose=True)
        pf = ecase.get_entang_profile()
        ecase.print_entang_profiles([pf], dm_bd.row_shape)

    main()

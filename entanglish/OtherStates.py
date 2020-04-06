import numpy as np
from entanglish.DenMat import *


class OtherStates:
    """
    This class has no constructor. It contains only static methods. Its
    methods return various quantum states which we decided were too simple
    to merit their own individual class.

    """

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
        print('bound entang state **********************')
        dm_bd = OtherStates.get_den_mat_with_bound_entang(.5)

        recursion_init = 'eigen+'
        num_ab_steps = 30
        ecase = SquashedEnt(dm_bd, num_ab_steps,
                            recursion_init=recursion_init, verbose=True)
        pf = ecase.get_entang_profile()
        ecase.print_entang_profiles([pf], dm_bd.row_shape)

    main()

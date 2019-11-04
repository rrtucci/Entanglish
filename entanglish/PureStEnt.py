from entanglish.EntangCase import *


class PureStEnt(EntangCase):
    """
    This class is a child of class EntangCase. Its purpose is to calculate
    the ( bipartite) quantum entanglement E_xy of a pure state |psi_xy>
    where x and y constitute a bi-partition of the set of all qudits.
    E_xy is defined here as the von Neumann entropy S(dm_x) of a density
    matrix dm_x, where dm_x = trace_y dm_xy, where dm_xy = |psi_xy><psi_xy|.

    Attributes
    ----------
    den_mat : DenMat

    """

    def __init__(self, den_mat, method='eigen',
                 num_bstrap_steps=1, check_purity=True, verbose=False):
        """
        Constructor. If check_purity = True, checks that den_mat is a pure
        state (has rank 1)

        Parameters
        ----------
        den_mat : DenMat
        method : str
        num_bstrap_steps : int
        check_purity : bool
        verbose : bool

        Returns
        -------


        """
        if check_purity:
            assert den_mat.is_pure_state(), \
                'the density matrix does not represent a pure state'
        EntangCase.__init__(self, len(den_mat.row_shape), method,
                        num_bstrap_steps, verbose)
        self.den_mat = den_mat

    def get_entang(self, axes_subset):
        """
        This method returns the von Neumann entropy S(dm_x), where dm_x =
        trace_y dm_xy, where x = axes_subset, and y is the set of all
        other axes.

        Parameters
        ----------
        axes_subset : set[int]

        Returns
        -------
        float

        """
        traced_axes_set = self.den_mat.get_set_of_all_other_axes(axes_subset)
        partial_dm = self.den_mat.get_partial_tr(traced_axes_set)
        if self.method == 'eigen':
            entang = partial_dm.get_entropy('eigen')
        elif self.method == 'pade':
            entang = partial_dm.get_entropy('pade')
        elif self.method == 'pert':
            if self.num_bstrap_steps == 1:
                pert = DenMatPertTheory.new_with_separable_dm0(partial_dm,
                                                               self.verbose)
                evas = pert.evas_of_dm_to_2nd_order
            else:
                evas, evec_cols = \
                    DenMatPertTheory.do_bstrap_with_separable_dm0(
                        partial_dm, self.num_bstrap_steps, self.verbose)
            if self.verbose:
                print('approx evas', np.sort(evas))
                print('exact evas', np.linalg.eigvalsh(partial_dm.arr))
            evas[evas < 1e-6] = 1e-6
            evas /= np.sum(evas)
            entang = ut.get_entropy_from_probs(evas)
        else:
            assert False
        return entang


if __name__ == "__main__":
    from entanglish.SymNupState import *

    def main():
        def extra_str(meth, num_steps):
            return ', ' + str(num_steps) + ' steps' \
                    if meth == 'pert' else ''
        num_bits = 4
        num_up = 2
        dm1 = DenMat(1 << num_bits, tuple([2]*num_bits))
        st = SymNupState(num_up, num_bits)
        st_vec = st.get_st_vec()
        dm1.set_arr_from_st_vec(st_vec)
        print('-------------------dm1')
        for method in ['eigen', 'pert']:
            num_bstrap_steps = 40
            print('-----method=' + method +
                  extra_str(method, num_bstrap_steps))
            ecase = PureStEnt(dm1, method,
                                 num_bstrap_steps, verbose=False)
            print('entang_023: algo value, known value\n',
                  ecase.get_entang({0, 2, 3}),
                  st.get_known_entang(3))
            print('entang_02: algo value, known value\n',
                  ecase.get_entang({0, 2}),
                  st.get_known_entang(2))
            print('entang_1: algo value, known value\n',
                  ecase.get_entang({1}),
                  st.get_known_entang(1))

        dm2 = DenMat(24, (3, 2, 2, 2))
        np.random.seed(123)
        st_vec = ut.random_st_vec(24)
        dm2.set_arr_from_st_vec(st_vec)
        print('-------------------dm2')
        num_bstrap_steps = 40
        for method in ['eigen', 'pert']:
            print('-----method=', method +
                  extra_str(method, num_bstrap_steps))
            ecase = PureStEnt(dm2, method,
                                 num_bstrap_steps, verbose=False)
            print('entang_023:', ecase.get_entang({0, 2, 3}))
            print('entang_02:', ecase.get_entang({0, 2}))
            print('entang_1:', ecase.get_entang({1}))
    main()

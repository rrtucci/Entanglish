from entanglish.DenMat import *
from entanglish.Entang import *
from entanglish.DenMatPertTheory import *


class EntangPureSt(Entang):
    """
    This class is a child of class Entang. Its purpose is to calculate the (
    bipartite) quantum entanglement E_xy of a pure state |psi_xy> where x
    and y are disjoint sets of qudits. E_xy is defined here as the von
    Neumann entropy S(dm_x) of a density matrix dm_x, where dm_x = trace_y
    dm_xy, where dm_xy = |psi_xy><psi_xy|.

    Attributes
    ----------
    den_mat : DenMat
        density matrix for a pure state. What we called dm_xy in the doc
        string for this class
    method : str
        method used to calculate the (natural) log of a density matrix that
        appears in the definition of the von Neumann entropy. This parameter
        can be either 'exact-eigen', 'exact-pade', '2nd-order'.
    num_bstrap_steps : int
        number bootstrap steps used in perturbation theory. Only used if
        method = '2nd-order'
    verbose : bool

    """

    def __init__(self, den_mat, method='exact-eigen',
                 num_bstrap_steps=1, verbose=False):
        """
        Constructor
        Checks that den_mat is a pure state (has rank 1)

        Parameters
        ----------
        den_mat : DenMat
        method : str
        num_bstrap_steps : int
        verbose : bool

        Returns
        -------


        """
        assert den_mat.is_pure_state(), \
            'the density matrix does not represent a pure state'
        Entang.__init__(self, len(den_mat.row_shape))
        self.den_mat = den_mat
        self.method = method
        assert method in ['exact-eigen', 'exact-pade', '2nd-order']
        self.num_bstrap_steps = num_bstrap_steps
        assert num_bstrap_steps > 0
        self.verbose = verbose

    def get_entang(self, traced_axes_set):
        """
        This method returns the von Neumann entropy S(dm_x), where dm_x =
        trace_y dm_xy, where x = traced_axes_set, and y is the set of all
        other axes.

        Parameters
        ----------
        traced_axes_set : set[int]

        Returns
        -------
        float

        """
        partial_dm = self.den_mat.get_partial_tr(traced_axes_set)
        if self.method == 'exact-eigen':
            entang = partial_dm.get_entropy('eigen')
        elif self.method == 'exact-pade':
            entang = partial_dm.get_entropy('pade')
        elif self.method == '2nd-order':
            if self.num_bstrap_steps == 1:
                pert = DenMatPertTheory.new_from_dm(partial_dm)
                evas = pert.evas_of_dm_to_2nd_order
            else:
                evas, evec_cols = DenMatPertTheory.get_bstrap_fin_eigen_sys_m(
                    partial_dm, self.num_bstrap_steps)
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
            if meth == '2nd-order':
                return ', num_bstrap_steps=' + str(num_steps)
            else:
                return ''
        num_bits = 4
        num_up = 2
        dm1 = DenMat(1 << num_bits, tuple([2]*num_bits))
        st = SymNupState(num_up, num_bits)
        st_vec = st.get_st_vec()
        dm1.set_arr_from_st_vec(st_vec)
        print('-------------------dm1')
        for method in ['exact-eigen', '2nd-order']:
            num_bstrap_steps = 10
            print('-----method=' + method +
                  extra_str(method, num_bstrap_steps))
            entang_pure = EntangPureSt(dm1, method,
                num_bstrap_steps, verbose=True)
            print('entang_023: calculated value, known value\n',
                  entang_pure.get_entang({0, 2, 3}),
                  st.get_entang(3))
            print('entang_02: calculated value, known value\n',
                  entang_pure.get_entang({0, 2}),
                  st.get_entang(2))
            print('entang_1: calculated value, known value\n',
                  entang_pure.get_entang({1}),
                  st.get_entang(1))

        dm2 = DenMat(24, (3, 2, 2, 2))
        np.random.seed(123)
        st_vec = ut.random_st_vec(24)
        dm2.set_arr_from_st_vec(st_vec)
        print('-------------------dm2')
        num_bstrap_steps = 10
        for method in ['exact-eigen', '2nd-order']:
            print('-----method=', method+
                  extra_str(method, num_bstrap_steps))
            entang_pure = EntangPureSt(dm2, method,
                         num_bstrap_steps, verbose=False)
            print('entang_023:', entang_pure.get_entang({0, 2, 3}))
            print('entang_02:', entang_pure.get_entang({0, 2}))
            print('entang_1:', entang_pure.get_entang({1}))
    main()

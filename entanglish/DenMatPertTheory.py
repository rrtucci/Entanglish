from entanglish.DenMat import *
import copy as cp


class DenMatPertTheory:
    """
    DenMat, dm, den_mat all stand for density matrix.

    This class performs operations associated with perturbation theory (
    more precisely, second order, time-independent perturbation theory in
    quantum mechanics) of the eigenvalues and eigenvectors of a density
    matrix.

    In quantum mechanics, such perturbation theory is used to approximate
    the eigenvalues and eigenvectors of a Hamiltonian H

    H = H0 + V,

    when the evas and evecs of H0 are known exactly. Pert theory only
    depends on the fact that H is Hermitian. Since a density matrix dm

    dm = dm0 + del_dm

    is Hermitian too, pert theory can be used for density matrices as well
    as for Hamiltonians. The question is, what to use for dm0? The
    constructor of this class (__init__) leaves that question unanswered.
    However, the static function DenMatPertTheory.new_with_separable_dm0()
    answers this question. It takes a density matrix dm as input,
    and returns an object of this class, i.e., DenMatPertTheory, assuming
    that dm0 equals the Kronecker product of the one-axis marginals of dm.

    The marginals of a square array arr is defined as a list of partial
    traces of arr. The n'th item in the list of marginals is the partial
    trace of arr, traced over all qudits except the n'th. The Kronecker
    product of the marginals of arr is a "separable" density matrix, in the
    sense that there is no correlation among its qudits.

    Each marginal is the density matrix of a qudit, so it is a d x d matrix,
    where d is the number of states of the qudit. If d is small for all the
    marginals, it is much easier to diagonalize every marginal than to
    diagonalize the whole density matrix dm. So it is reasonable to assume
    that the evas and evecs of dm0 can be calculated easily exactly. We wish
    to use those evas and evecs to approximate perturbatively the evas and
    evecs of dm.

    We will call an eigensystem of a density matrix: a tuple, whose first
    item is a 1D numpy array, call it evas, with the eigenvalues of the
    density matrix, and the second item is a 2D numpy array, call it
    eigen_cols, whose i'th columns is an eigenvector for the i'th eigenvalue
    (i.e., evas[i]) of the density matrix.

    See Ref.1 for a more detailed explanation of the algos used in this class.

    References
    ----------
    1. R.R. Tucci, "A New  Algorithm for Calculating Squashed Entanglement
    and a Python Implementation Thereof"

    Attributes
    ----------
    del_dm : DenMat
        defined above
    del_dm_in_sbasis: DenMat
        del_dm is in inbasis. It is convenient to change it to sbasis (
        separable basis) so that if v1 = dm0_eigen_sys[1][n1] and v2 =
        dm0_eigen_sys[1][n2] then del_dm_in_sbasis[n1, n2] = <v1| del_dm |v2>
    dm0_eigen_sys : tuple[np.ndarray, np.ndarray]
        eigensystem of density matrix dm0.
    evas_of_dm_to_2nd_order : np.ndarray
        1D array of floats. Eigenvalues of dm to second order.
    evec_cols_of_dm_to_2nd_order : np.ndarray
        This is a unitary matrix with (a second order approx of) the
        eigenvectors of dm as columns. If this matrix is U, the `dm \approx
        UDU^dag`, where D is diagonal and U^dag is the Hermitian conjugate of
        U.
    verbose : bool

    """

    def __init__(self, dm0_eigen_sys, del_dm, verbose=False):
        """
        Constructor
        
        Parameters
        ----------
        dm0_eigen_sys : tuple[np.ndarray, np.ndarray]
        del_dm : DenMat
        verbose : bool

        Returns
        -------


        """
        self.verbose = verbose
        self.dm0_eigen_sys = dm0_eigen_sys
        # will not assume trace is one
        # ut.assert_is_prob_dist(np.array(dm0_eigen_sys[0]), halt=True)
        self.del_dm = del_dm
        # even if dm and dm0 don't have unit trace,
        # assume their traces are equal
        assert abs(del_dm.trace()) < 1e-5, abs(del_dm.trace())
        # print('aaaaaaaa1', np.linalg.norm(del_dm.arr))
        self.del_dm_in_sbasis = del_dm.switch_arr_basis(dm0_eigen_sys[1])
        # print('aaaaaaaa2', np.linalg.norm(self.del_dm_in_sbasis.arr))
        # print('aaaaaaa2-zero-if-esys-unitary', np.linalg.norm(np.dot(
        #    dm0_eigen_sys[1].conj().T, dm0_eigen_sys[1])-
        #       np.eye(del_dm.num_rows)))
        self.diagonalize_del_dm_in_sbasis_in_degenerate_spaces()
        # print('aaaaaaaa3', np.linalg.norm(self.del_dm_in_sbasis.arr))

        self.evas_of_dm_to_2nd_order = None
        self.evec_cols_of_dm_to_2nd_order = None

        self.set_evas_of_dm_to_2nd_order()
        self.set_evec_cols_of_dm_to_2nd_order()

    @staticmethod
    def new_with_separable_dm0(dm, verbose=False):
        """
        This method returns a DenMatPertTheory built from a density matrix
        dm, assuming that dm0 is the Kronecker product of the marginals of dm.

        Parameters
        ----------
        dm : DenMat
        verbose : bool

        Returns
        -------
        DenMatPertTheory

        """
        if dm.marginals is None:
            dm.set_marginals()
        esys = dm.get_eigen_sys_of_marginals()
        dm0_eigen_sys = (ut.kron_prod(esys[0]), ut.kron_prod(esys[1]))
        arr = ut.kron_prod([marg.arr for marg in dm.marginals])
        dm0 = DenMat(dm.num_rows, dm.row_shape, arr)

        return DenMatPertTheory(dm0_eigen_sys, dm - dm0, verbose)

    def get_dm0(self):
        """
        This method returns the unperturbed density matrix dm0. We define
        this method so as to avoid storing dm0.

        Returns
        -------
        DenMat

        """
        evas = self.dm0_eigen_sys[0]
        evec_cols = self.dm0_eigen_sys[1]
        arr = ut.fun_of_herm_arr_from_eigen_sys(lambda x: x, evas, evec_cols)
        num_rows = self.del_dm.num_rows
        row_shape = self.del_dm.row_shape
        return DenMat(num_rows, row_shape, arr)

    def diagonalize_del_dm_in_sbasis_in_degenerate_spaces(self):
        """
        This function performs a similarity transformation on
        del_dm_in_sbasis so that it is diagonal on the subspaces of
        eigenvectors with degerate eigenvalues of dm0. After this
        transformation, if j != k and evas[j] = evas[k] then
        del_dm_in_sbasis[j, k] = 0, where evas are the eigenvalues of dm0.
        This is called degenerate 2nd order perturbation theory.

        Returns
        -------
        None

        """
        classes = ut.get_equiv_classes(list(self.dm0_eigen_sys[0]))
        umat = np.eye(self.del_dm.num_rows, dtype=complex)
        # eq = equivalence
        for eq_class in classes:
            dim = len(eq_class)
            if dim > 1:
                if self.verbose:
                    print("non-trivial eq class", eq_class)
                eq_class = sorted(eq_class)
                arr = np.empty((dim, dim), dtype=complex)
                for k1, kk1 in enumerate(eq_class):
                    for k2, kk2 in enumerate(eq_class):
                        arr[k1, k2] = self.del_dm_in_sbasis[kk1, kk2]
                norm_arr = np.linalg.norm(arr)
                # print('norm arr', norm_arr)
                if norm_arr > 1e-4:
                    _, evec_cols = np.linalg.eigh(arr)
                    assert ut.is_unitary_arr(evec_cols)
                    # print('bbbbbbbb', np.around(np.dot(np.dot(
                    #         evec_cols.conj().T,
                    #               arr), evec_cols).real,4))
                    for k1, kk1 in enumerate(eq_class):
                        for k2, kk2 in enumerate(eq_class):
                            umat[kk1, kk2] = evec_cols[k1, k2]
        assert ut.is_unitary_arr(umat)
        # print('ccccccccnorm of rotated_del_dm', np.linalg.norm(
        # self.del_dm_in_sbasis.arr))
        rotated_del_dm = np.dot(
            np.dot(umat.conj().T, self.del_dm_in_sbasis.arr), umat)
        # print('norm of rotated_del_dm', np.linalg.norm(rotated_del_dm))
        self.del_dm_in_sbasis.arr = rotated_del_dm

    def set_evas_of_dm_to_2nd_order(self):
        """
        This function sets the class attribute evas_of_dm_to_2nd_order (the
        eigenvalues of dm, to second order in pert theory). Actually,
        since it's not that difficult to go from 2nd to 3rd order
        perturbation for eigenvalues (for the eigenvectors it is more
        difficult), we calculate the 3rd order approx for evas.

        Formulas used here come from the Wikipedia article Ref.1. That
        article has a figure containing the eigenvalues and eigenvectors to
        fifth order in perturbation theory!

        References
        ----------
        1. `<https://en.wikipedia.org/wiki/Perturbation_theory_(
        quantum_mechanics)>`_

        Returns
        -------
        None

        """
        num_evas = len(self.dm0_eigen_sys[0])
        evas_to_2nd_order = []
        use_3rd_order = True
        for n in range(num_evas):
            eva = self.dm0_eigen_sys[0][n]
            # print('...../', eva, self.del_dm.arr)
            eva += self.del_dm_in_sbasis[n, n].real
            for k1 in range(num_evas):
                if k1 != n:
                    me_n_k1 = self.del_dm_in_sbasis[n, k1]
                    me_k1_n = self.del_dm_in_sbasis[k1, n]
                    me_n_n = self.del_dm_in_sbasis[n, n]
                    lam_n_k1 = self.dm0_eigen_sys[0][n] -\
                               self.dm0_eigen_sys[0][k1]
                    if abs(lam_n_k1) > 1e-5:
                        eva += (me_n_k1*me_k1_n).real/lam_n_k1
                        if use_3rd_order:
                            eva += -(me_n_n*me_n_k1*me_k1_n).real/lam_n_k1**2
                    else:
                        # if denominator is zero, numerator should be too.
                        # if it isn't, del_dm must be diagonalized
                        #  over the degenerate eigenspace so that the
                        # numerator becomes zero. This is called degenerate
                        # 2nd order perturbation theory

                        # assert abs(me_n_k1) < 1e-5, str(me_n_k1) + \
                        #                         ' / ' + str(abs(lam_n_k1))
                        pass
                    if use_3rd_order:
                        for k2 in range(num_evas):
                            if k2 != n:
                                me_k2_k1 = self.del_dm_in_sbasis[k2, k1]
                                me_n_k2 = self.del_dm_in_sbasis[n, k2]
                                lam_n_k2 = self.dm0_eigen_sys[0][n] \
                                           - self.dm0_eigen_sys[0][k2]
                                numer = (me_n_k2*me_k2_k1*me_k1_n).real
                                denom = lam_n_k1*lam_n_k2
                                if abs(denom) > 1e-6:
                                    # print('.....//',
                                    # n, k1, k2, lam_n_k1, lam_n_k2)
                                    eva += numer/denom
                                    # print(';;;', eva)
                                else:
                                    pass
                                    # assert abs(numer) < 1e-5, str(numer)

            evas_to_2nd_order.append(eva)

        evas = np.array(evas_to_2nd_order)

        fix = True
        if fix:
            evas = np.array([ut.clip(x, [0, 1]) for x in evas])
            evas = evas/sum(evas)

        self.evas_of_dm_to_2nd_order = evas
        # print("===", evas_to_2nd_order)

    def set_evec_cols_of_dm_to_2nd_order(self):
        """
        This function sets the class attribute evec_cols_of_dm_to_2nd_order
        (a matrix with the eigenvectors, as columns, of dm, to second order
        in pert theory).

        Formulas used here come from the Wikipedia article Ref.1. That
        article has a figure containing the eigenvalues and eigenvectors to
        fifth order in perturbation theory!

        References
        ----------
        1. https://en.wikipedia.org/wiki/Perturbation_theory_(
        quantum_mechanics)

        Returns
        -------
        None
        """
        num_evas = len(self.dm0_eigen_sys[0])
        coef_n = np.zeros((num_evas,), dtype=complex)
        coef_n_k1 = np.zeros((num_evas, num_evas), dtype=complex)
        for n in range(num_evas):
            for k1 in range(num_evas):
                if k1 != n:
                    # me = matrix element, lam = eigenvalue lambda
                    me_k1_n = self.del_dm_in_sbasis[k1, n]
                    me_n_k1 = self.del_dm_in_sbasis[n, k1]
                    me_n_n = self.del_dm_in_sbasis[n, n]
                    lam_n_k1 = self.dm0_eigen_sys[0][n] - \
                               self.dm0_eigen_sys[0][k1]
                    if abs(lam_n_k1) > 1e-6:
                        coef_n_k1[n, k1] += me_k1_n/lam_n_k1\
                            - me_n_n*me_k1_n/lam_n_k1**2
                        coef_n[n] += -(1/2)*me_n_k1*me_k1_n/lam_n_k1**2
                        for k2 in range(num_evas):
                            if k2 != n:
                                me_k1_k2 = self.del_dm_in_sbasis[k1, k2]
                                me_k2_n = self.del_dm_in_sbasis[k2, n]
                                lam_n_k2 = self.dm0_eigen_sys[0][n] - \
                                           self.dm0_eigen_sys[0][k2]
                                if abs(lam_n_k2) > 1e-6:
                                    coef_n_k1[n, k1] += \
                                        me_k1_k2*me_k2_n/(lam_n_k1*lam_n_k2)
                                else:
                                    # assert abs(me_k2_n) < 1e-
                                    pass
                    else:
                        # if denominator is zero, numerator should be too.
                        # if it isn't, del_dm must be diagonalized
                        #  over the degenerate eigenspace so that the
                        # numerator becomes zero. This is called degenerate
                        # 2nd order perturbation theory

                        # assert abs(me_k1_n) < 1e-6
                        pass
        # umat = unitary matrix
        # umat0 contains evecs as cols of separable den mat to 0th order
        umat0 = self.dm0_eigen_sys[1]
        umat = cp.copy(umat0)
        # print('---------..,,xx', num_evas, umat.shape)
        for n in range(num_evas):
            umat[:, n] += coef_n[n]*umat0[:, n]
            for k1 in range(num_evas):
                umat[:, n] += coef_n_k1[n, k1]*umat0[:, k1]

        fix = True
        if fix:
            # this use of the qr decomposition replaces umat by a matrix
            # which is close to it, but repaired to fix any
            # deviation from unitarity
            umat, _ = np.linalg.qr(umat)

        self.evec_cols_of_dm_to_2nd_order = umat

    @staticmethod
    def do_bstrap(dm0_eigen_sys, del_dm, num_steps=1, verbose=False):
        """
        If we define sub_del_dm = del_dm/num_steps, then this method
        calculates an eigensystem for dm0 + sub_del_dm*(k+1) from the
        eigensystem for dm0 + sub_del_dm*k for k = 0, 1, 2, ...,
        num_steps-1. This produces a "bootstrap sequence of eigensystems"
        that is num_steps steps long. This method returns the final
        eigensystem of that bootstrap sequence.

        Parameters
        ----------
        dm0_eigen_sys : tuple[np.ndarray, np.ndarray]
        del_dm : DenMat
        num_steps : int
        verbose : bool

        Returns
        -------
        tuple[np.ndaray, np.ndarray]

        """
        sub_del_dm = del_dm * (1/num_steps)
        # print('xxxxxxxxxxxxxx-sub-del-dm\n', sub_del_dm)
        if verbose:
            print('------------------beginning of step', 0, '/', num_steps)
            print('bstrap input evas\n', np.sort(dm0_eigen_sys[0]),
                  'sum=', np.sum(dm0_eigen_sys[0]))
        cur_pert = DenMatPertTheory(dm0_eigen_sys, sub_del_dm, verbose)

        for k in range(1, num_steps):
            if verbose:
                print('------------------beginning of step', k, '/', num_steps)
                print('bstrap input evas\n', np.sort(
                    cur_pert.evas_of_dm_to_2nd_order),
                    'sum=', np.sum(cur_pert.evas_of_dm_to_2nd_order))
            cur_dm0_eigen_sys = (cur_pert.evas_of_dm_to_2nd_order,
                                 cur_pert.evec_cols_of_dm_to_2nd_order)
            cur_pert = DenMatPertTheory(cur_dm0_eigen_sys, sub_del_dm, verbose)
        evas = cur_pert.evas_of_dm_to_2nd_order
        evec_cols = cur_pert.evec_cols_of_dm_to_2nd_order
        return evas, evec_cols

    @staticmethod
    def do_bstrap_with_separable_dm0(dm, num_steps=1, verbose=False):
        """
        This method returns the same thing as the method (found in its
        parent class) DenMatPertTheory.do_bstrap( ). However, their names
        differ by a '_with_separable_dm0' at the end and their inputs are
        different. This one takes as input a density matrix dm and
        calculates dm0_eigen_sys and del_dm from that, assuming that dm0 is
        the Kronecker product of the marginals of dm.

        Parameters
        ----------
        dm : DenMat
        num_steps : int
        verbose : bool

        Returns
        -------
        tuple[np.ndaray, np.ndarray]

        """
        if dm.marginals is None:
            dm.set_marginals()
        esys = dm.get_eigen_sys_of_marginals()
        dm0_eigen_sys = (ut.kron_prod(esys[0]),
                         ut.kron_prod(esys[1]))
        arr = ut.kron_prod([marg.arr for marg in dm.marginals])
        dm0 = DenMat(dm.num_rows, dm.row_shape, arr)
        return DenMatPertTheory.do_bstrap(
            dm0_eigen_sys, dm-dm0, num_steps, verbose)


if __name__ == "__main__":
    from entanglish.SymNupState import *

    def main_test(dm, exact_evas, pert, num_steps):
        """

        Parameters
        ----------
        dm : DenMat
        exact_evas : np.ndarray
        pert : DenMatPertTheory
        num_steps : int

        Returns
        -------
        None

        """
        # print('******do ', num_steps, ' steps:')
        fin_esys = DenMatPertTheory.do_bstrap(
            pert.dm0_eigen_sys, pert.del_dm, num_steps, pert.verbose)
        print('final evas of dm to 2nd order\n', np.sort(fin_esys[0]),
              'sum=', np.sum(fin_esys[0]))
        print('evas_of_dm\n', np.sort(exact_evas), 'sum=', np.sum(exact_evas))

        dm_2nd_order = DenMat.get_fun_of_dm_from_eigen_sys(
            dm.num_rows, dm.row_shape, fin_esys, lambda x: x)
        # print('dm_to_2nd_order\n', dm_2nd_order)
        diff_arr = dm_2nd_order.arr - dm.arr
        # print('dm_to_2nd_order - dm\n', diff_arr)
        print('distance(dm_to_2nd_order, dm)\n', np.linalg.norm(diff_arr))

        true_exp = dm.exp()
        approx_exp = DenMat.get_fun_of_dm_from_eigen_sys(
            dm.num_rows, dm.row_shape, fin_esys, np.exp)
        print('distance(approx_exp, true_exp)\n',
              np.linalg.norm(approx_exp.arr - true_exp.arr))

    def main1():
        print('--------------------main1')
        np.random.seed(123)
        dm = DenMat(8, (2, 2, 2))
        exact_evas = np.array([.1, .3, .3, .1, .2, .1, .6, .7])
        exact_evas /= np.sum(exact_evas)
        print('evas of dm\n', np.sort(exact_evas))
        dm.set_arr_to_rand_den_mat(exact_evas)

        pert = DenMatPertTheory.new_with_separable_dm0(dm, verbose=True)
        # print('evas of dm to 2nd order (after 1 step)\n',
        #       np.sort(pert.evas_of_dm_to_2nd_order),
        #       'sum=', np.sum(pert.evas_of_dm_to_2nd_order))
        # print('evas of dm\n', np.sort(evas))

        # print('evas_of_dm0\n', pert.dm0_eigen_sys[0])
        #
        # print('diag of del_dm in sbasis\n',
        #       np.diag(pert.del_dm_in_sbasis.arr.real))

        # print('del_dm\n', pert.del_dm)
        # dm0 = pert.get_dm0()
        # print('dm0\n', dm0)

        num_steps = 40
        # print('evas_of_dm\n', evas)
        main_test(dm, exact_evas, pert, num_steps)

    def main2():
        print('--------------------main2')
        np.random.seed(123)

        dm1 = DenMat(3, (3,))
        evas1 = np.array([.1, .1, .4])
        evas1 /= np.sum(evas1)
        dm1.set_arr_to_rand_den_mat(evas1)
        # print('dm1\n', dm1)

        dm2 = DenMat(2, (2,))
        evas2 = np.array([.8, .3])
        evas2 /= np.sum(evas2)
        dm2.set_arr_to_rand_den_mat(evas2)
        # print('dm2\n', dm2)

        dm = DenMat.get_kron_prod_of_den_mats([dm1, dm2])
        const = 0
        # const = 1
        dm.add_const_to_diag_of_arr(const)
        dm.normalize_diag_of_arr()
        print('kron evas\n', np.sort(ut.kron_prod([evas1, evas2])))
        exact_evas = np.linalg.eigvalsh(dm.arr)
        print('evas_of_dm\n', exact_evas)

        pert = DenMatPertTheory.new_with_separable_dm0(dm, verbose=True)

        # print('evas of dm to 2nd order (after 1 step)\n',
        #       np.sort(pert.evas_of_dm_to_2nd_order),
        #       'sum=', np.sum(pert.evas_of_dm_to_2nd_order))
        num_steps = 10
        main_test(dm, exact_evas, pert, num_steps)

    def main3():
        print('--------------------main3')
        num_qbits = 4
        num_up = 2
        dm = DenMat(1 << num_qbits, tuple([2]*num_qbits))
        st = SymNupState(num_up, num_qbits)
        st_vec = st.get_st_vec()
        dm.set_arr_from_st_vec(st_vec)
        # dm.depurify(.1)

        exact_evas = np.linalg.eigvalsh(dm.arr)
        print('evas_of_dm\n', exact_evas, 'sum=', np.sum(exact_evas))

        pert = DenMatPertTheory.new_with_separable_dm0(dm, verbose=True)
        num_steps = 80
        main_test(dm, exact_evas, pert, num_steps)

    main1()
    main2()
    main3()



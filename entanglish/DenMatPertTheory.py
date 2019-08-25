from entanglish.DenMat import *
import copy as cp


class DenMatPertTheory:
    """
    DenMat, dm, den_mat all stand for density matrix.

    This class performs operations associated with perturbation theory (
    second order, mainly) of the eigenvalues and eigenvectors of a density
    matrix.

    In quantum mechanics, perturbation theory is used to approximate the
    eigenvalues and eigenvectors of a Hamiltonian H

    H = H0 + V,

    when the evas and evecs of H0 are known exactly. Pert theory only
    depends on the fact that H is Hermitian. Since a density matrix dm

    dm = dm0 + del_dm

    is Hermitian too, pert theory can be used for density matrices as well
    as for Hamiltonians. The question is, what to use for dm0? The
    constructor of this class (__init__) leaves that question unanswered.
    However, the static function DenMatPertTheory.new_from_dm() answers this
    question. It takes a density matrix dm as input, and returns an object
    of this class, i.e., DenMatPertTheory, assuming that dm0 equals the
    Kronecker product of the marginals of dm.

    The marginals of a square array arr is defined as a list of partial
    traces of arr. The n'th item in the list of marginals is the partial
    trace of arr, traced over all qudits except the n'th. The Kronecker
    product of the marginals of arr is a "separable" density matrix, in the
    sense that there is no correlation among its qudits.

    Each marginal is the density matrix of a qudit, so it is a d x d matrix,
    where d is the number of states of the qudit. If d is small for all the
    marginals, it is much easier to diagonalize every marginal than to
    diagonalize the whole density matrix dm. So it is reasonable to assume
    that the evas and evecs of dm0 can be calculated easily exactly,
    and that we wish to use those to approximate perturbatively the evas and
    evecs of dm.

    We will call an eigensystem of a density matrix: a tuple, whose first
    item is a 1D numpy array, call it evas, with the eigenvalues of the
    density matrix, and the second item is a 2D numpy array, call it
    eigen_cols, whose i'th columns is an eigenvector for the i'th eigenvalue
    (i.e., evas[i]) of the density matrix.


    Attributes
    ----------
    __evec_cols_of_dm_to_2nd_order : np.ndarray
        This is a unitary matrix with (a second order approx of) the
        eigenvectors of dm as columns. If this matrix is U, the dm \approx
        UDU^dag, where D is diagonal and U^dag is the Hermitian conjugate of
        U.
    del_dm : DenMat
        defined above
    del_dm_in_sbasis: DenMat
        del_dm is in inbasis. It is convenient to change it to sbasis (
        separable basis) so that if v1 = dm0_eigen_sys[0][n1] and v2 =
        dm0_eigen_sys[0][n2] then del_dm_in_sbasis[n1, n2] = <v1| del_dm |v2>
    dm0_eigen_sys : tuple[np.ndarray, np.ndarray]
        eigensystem of density matrix dm0.
    evas_of_dm_to_2nd_order : np.ndarray
        1D array of floats. Eigenvalues of dm to second order.

    """

    def __init__(self, dm0_eigen_sys, del_dm):
        """
        Constructor
        
        Parameters
        ----------
        dm0_eigen_sys : tuple[np.ndarray, np.ndarray]
        del_dm : DenMat

        Returns
        -------


        """
        self.dm0_eigen_sys = dm0_eigen_sys
        self.del_dm = del_dm
        self.del_dm_in_sbasis = del_dm.switch_arr_basis(dm0_eigen_sys[1])
        self.diagonalize_del_dm_in_sbasis_in_degenerate_spaces()

        self.evas_of_dm_to_2nd_order = None
        self.__evec_cols_of_dm_to_2nd_order = None

        self.set_evas_of_dm_to_2nd_order()
        # calculate self.evec_cols_of_dm_to_2nd_order
        # only if it is needed

    @staticmethod
    def new_from_dm(dm):
        """
        This method returns a DenMatPertTheory built from a density matrix
        dm, assuming that dm0 is the Kronecker product of the marginals of dm.

        Parameters
        ----------
        dm : DenMat

        Returns
        -------
        DenMatPertTheory

        """
        esys = dm.get_eigen_sys_of_marginals()
        dm0_eigen_sys = (ut.kron_prod(esys[0]), ut.kron_prod(esys[1]))
        arr = ut.fun_of_herm_arr_from_eigen_sys(
            lambda x: x, dm0_eigen_sys[0], dm0_eigen_sys[1])
        dm0 = DenMat(dm.num_rows, dm.row_shape, arr)

        return DenMatPertTheory(dm0_eigen_sys, dm - dm0)

    @property
    def evec_cols_of_dm_to_2nd_order(self):
        """
        This method calculates the class attribute
        self.__evec_cols_of_dm_to_2nd_order if it is empty and returns it
        when self.evec_cols_of_dm_to_2nd_order is called.

        Returns
        -------
        np.ndarray

        """
        if self.__evec_cols_of_dm_to_2nd_order is None:
            self.set_evec_cols_of_dm_to_2nd_order()
        return self.__evec_cols_of_dm_to_2nd_order

    @evec_cols_of_dm_to_2nd_order.setter
    def evec_cols_of_dm_to_2nd_order(self, x):
        """
        Setter. It permits
        self.evec_cols_of_dm_to_2nd_order = x

        Parameters
        ----------
        x : np.ndarray

        Returns
        -------
        None

        """
        self.__evec_cols_of_dm_to_2nd_order = x

    def get_dm0(self):
        """
        This method returns the unperturbed density matrix dm0.

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
                eq_class = sorted(eq_class)
                arr = np.empty((dim, dim), dtype=complex)
                for k1, kk1 in enumerate(eq_class):
                    for k2, kk2 in enumerate(eq_class):
                        arr[k1, k2] = self.del_dm_in_sbasis[kk1, kk2]
                _, evec_cols = np.linalg.eigh(arr)
                for k1, kk1 in enumerate(eq_class):
                    for k2, kk2 in enumerate(eq_class):
                        umat[kk1, kk2] = evec_cols[k1, k2]
        self.del_dm_in_sbasis.arr = \
            np.dot(np.dot(umat.conj().T, self.del_dm_in_sbasis.arr), umat)

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
        1. https://en.wikipedia.org/wiki/Perturbation_theory_(
        quantum_mechanics)

        Returns
        -------
        None

        """
        num_evas = len(self.dm0_eigen_sys[0])
        evas_to_2nd_order = []
        use_3rd_order= True
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
                    if abs(lam_n_k1) > 1e-6:
                        eva += (me_n_k1*me_k1_n).real/lam_n_k1
                        if use_3rd_order:
                            eva += -(me_n_n*me_n_k1*me_k1_n).real/lam_n_k1**2
                    else:
                        # if denominator is zero, numerator should be too.
                        # if it isn't, del_dm must be diagonalized
                        #  over the degenerate eigenspace so that the
                        # numerator becomes zero. This is called degenerate
                        # 2nd order perturbation theory
                        assert abs(me_n_k1) < 1e-6
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
                                    assert abs(numer) < 1e-6

            evas_to_2nd_order.append(eva)
        self.evas_of_dm_to_2nd_order = np.array(evas_to_2nd_order)
        # print("===", evas_to_2nd_order)

    def set_evec_cols_of_dm_to_2nd_order(self):
        """
        This function sets the class attribute
        __evec_cols_of_dm_to_2nd_order (a matrix with the eigenvectors,
        as columns, of dm, to second order in pert theory).

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
                                    assert abs(me_k2_n) < 1e-6
                    else:
                        # if denominator is zero, numerator should be too.
                        # if it isn't, del_dm must be diagonalized
                        #  over the degenerate eigenspace so that the
                        # numerator becomes zero. This is called degenerate
                        # 2nd order perturbation theory
                        assert abs(me_k1_n) < 1e-6
        # umat = unitary matrix
        # umat0 contains evecs as cols of separable den mat to 0th order
        umat0 = self.dm0_eigen_sys[1]
        umat = cp.copy(umat0)
        # print('---------..,,xx', num_evas, umat.shape)
        for n in range(num_evas):
            umat[:, n] += coef_n[n]*umat0[:, n]
            for k1 in range(num_evas):
                umat[:, n] += coef_n_k1[n, k1]*umat0[:, k1]

        self.__evec_cols_of_dm_to_2nd_order = umat

    @staticmethod
    def get_bstrap_fin_eigen_sys(dm0_eigen_sys, del_dm, num_steps=1):
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

        Returns
        -------
        tuple[np.ndaray, np.ndarray]

        """
        sub_del_dm = del_dm * (1/num_steps)
        cur_pert = DenMatPertTheory(dm0_eigen_sys, sub_del_dm)
        for k in range(1, num_steps):
            cur_dm0_eigen_sys = (cur_pert.evas_of_dm_to_2nd_order,
                                 cur_pert.evec_cols_of_dm_to_2nd_order)
            cur_pert = DenMatPertTheory(cur_dm0_eigen_sys, sub_del_dm)
        evas = cur_pert.evas_of_dm_to_2nd_order
        evec_cols = cur_pert.evec_cols_of_dm_to_2nd_order
        return evas, evec_cols

    @staticmethod
    def get_bstrap_fin_eigen_sys_m(dm, num_steps=1):
        """
        This method returns the same thing as the method (found in its
        parent class) DenMatPertTheory.get_final_eigen_sys( ). However,
        their names differ by a '_m' at the end (_m stands for marginals)
        and their inputs are different. This one takes as input a density
        matrix dm and calculates dm0_eigen_sys and del_dm from that,
        assuming that dm0 is the Kronecker product of the marginals of dm.

        To get just an approx to dm instead of an approx to fun of dm, use
        fun_of_scalars = lambda x: x

        Parameters
        ----------
        dm : DenMat
        num_steps : int

        Returns
        -------
        tuple[np.ndaray, np.ndarray]

        """
        esys = dm.get_eigen_sys_of_marginals()
        dm0_eigen_sys = (ut.kron_prod(esys[0]),
                         ut.kron_prod(esys[1]))
        arr = ut.fun_of_herm_arr_from_eigen_sys(
            lambda x: x, dm0_eigen_sys[0], dm0_eigen_sys[1])
        dm0 = DenMat(dm.num_rows, dm.row_shape, arr)
        return DenMatPertTheory.get_bstrap_fin_eigen_sys(
            dm0_eigen_sys, dm-dm0, num_steps)

    @staticmethod
    def get_fun_of_dm_to_2nd_order(num_rows, row_shape,
            final_eigen_sys, fun_of_scalars):
        """
        If evas, U = final_eigen_sys and fun = fun_of_scalars, then this
        method returns U.fun(evas).U^dag, where U^dag is the Hermitian
        conjugate of the unitary matrix U.

        The function calculated (for example, np.exp, np.log, etc.) is
        passed in as the input 'fun_of_scalars'. To get just an approx to dm
        instead of an approx to fun of dm, use fun_of_scalars = lambda x: x

        Parameters
        ----------
        num_rows : int
        row_shape : tuple[int]
        final_eigen_sys : tuple[np.ndarray, np.ndarray]
        fun_of_scalars :
            function that can act on scalars or numpy arrays element-wise

        Returns
        -------
        DenMat

        """
        evas = final_eigen_sys[0]
        evec_cols = final_eigen_sys[1]
        arr = ut.fun_of_herm_arr_from_eigen_sys(
            fun_of_scalars, evas, evec_cols)
        return DenMat(num_rows, row_shape, arr)

if __name__ == "__main__":

    def main_test(dm, pert, num_steps):
        """

        Parameters
        ----------
        dm : DenMat
        pert : DenMatPertTheory
        num_steps : int

        Returns
        -------
        None

        """
        print('******after ', num_steps, ' steps:')
        fin_esys = DenMatPertTheory.get_bstrap_fin_eigen_sys(
            pert.dm0_eigen_sys, pert.del_dm, num_steps)
        print('evas of dm to 2nd order\n', fin_esys[0])
        dm_2nd_order = DenMatPertTheory.get_fun_of_dm_to_2nd_order(
            dm.num_rows, dm.row_shape, fin_esys, lambda x: x)
        # print('dm_to_2nd_order\n', dm_2nd_order)
        diff_arr = dm_2nd_order.arr - dm.arr
        # print('dm_to_2nd_order - dm\n', diff_arr)
        print('distance(dm_to_2nd_order, dm)\n', np.linalg.norm(diff_arr))

        true_exp = dm.exp()
        approx_exp = DenMatPertTheory.get_fun_of_dm_to_2nd_order(
            dm.num_rows, dm.row_shape, fin_esys, np.exp)
        print('distance(approx_exp, true_exp)\n',
              np.linalg.norm(approx_exp.arr - true_exp.arr))

    def main1():
        print('--------------------main1')
        np.random.seed(123)
        dm = DenMat(4, (2, 2))
        evas = np.array([.1, .3, .4, .5])
        evas /= np.sum(evas)
        print('evas of dm\n', evas)
        dm.set_arr_to_rand_den_mat(evas)

        pert = DenMatPertTheory.new_from_dm(dm)
        print('evas of dm to 2nd order (after 1 step)\n',
              pert.evas_of_dm_to_2nd_order,
              'sum=', np.sum(pert.evas_of_dm_to_2nd_order))

        # print('evas_of_dm0\n', pert.dm0_eigen_sys[0])
        #
        # print('diag of del_dm in sbasis\n',
        #       np.diag(pert.del_dm_in_sbasis.arr.real))

        # print('del_dm\n', pert.del_dm)
        # dm0 = pert.get_dm0()
        # print('dm0\n', dm0)

        num_steps = 10
        # print('evas_of_dm\n', evas)
        main_test(dm, pert, num_steps)

    def main2():
        print('--------------------main2')
        np.random.seed(123)

        dm1 = DenMat(3, (3,))
        evas1 = np.array([.1, .1, .4])
        evas1 /= np.sum(evas1)
        dm1.set_arr_to_rand_den_mat(evas1)
        print('dm1\n', dm1)

        dm2 = DenMat(2, (2,))
        evas2 = np.array([.1, .3])
        evas2 /= np.sum(evas2)
        dm2.set_arr_to_rand_den_mat(evas2)
        print('dm2\n', dm2)

        dm = DenMat.get_kron_prod_of_den_mats([dm1, dm2])

        dm.add_const_to_diag_of_arr(1e-1)
        dm.normalize_diag_of_arr()

        pert = DenMatPertTheory.new_from_dm(dm)

        num_steps = 1
        print('evas_of_dm\n', ut.kron_prod([evas1, evas2]))
        main_test(dm, pert, num_steps)

    main1()
    main2()



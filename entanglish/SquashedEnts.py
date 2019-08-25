from entanglish.Entang import *
from entanglish.DenMat import *
from entanglish.DenMatPertTheory import *
import numpy as np


class SquashedEnts(Entang):
    """
    This class is a child of class Entang. It's name is a pun on "Squashed
    Ants" and "Squashed Entanglements". Please laugh. It's purpose is to
    calculate the (bipartite) quantum entanglement E_xy of a mixed state
    density matrix dm_xy. E_xy is defined here as the squashed entanglement
    (see Wikipedia Ref. 1 for more detailed description and original
    references of squashed entanglement)

    E_xy = (1/2)*min[S(x : y | h)]

    where S(x : y | h) is the conditional mutual information (CMI,
    pronounced "see me") for a density matrix dm_xyh. The minimum is over all
    dm_xyh such that dm_xy = trace_h dm_xyh, where dm_xy is given and fixed.

    The squashed entanglement reduces to (1/2)*[S(x) + S(y)] = S(x) = S(y)
    when dm_xy is a pure state. This is precisely the definition of
    entanglement for pure states that we use in the class EntangPureSt.

    In this class, to calculate squashed entanglement, I (rrtucci) use an
    algorithm which is new, has never been published before by me or anyone
    else. It's based on a generalization to density matrices of the Arimoto
    Blahut (AB) algorithm. AB is used in classical information theory to
    calculate channel capacities. In a previous paper (see Ref. 2),
    I proposed an earlier algo for calculating squashed entanglement,
    also based on the AB algorithm. The method used here is an improvement
    of my earlier algo.

    References
    ----------
    1. https://en.wikipedia.org/wiki/Squashed_entanglement

    2. R.R. Tucci, "Relaxation Method For Calculating Quantum Entanglement"
    https://arxiv.org/abs/quant-ph/0101123


    Attributes
    ----------
    Kxy : DenMat          
        The input density matrix den_mat, after permuting its axes so that 
        the axes are given in the precise order self.x_axes + self.y_axes. 
        Kxy is referred to as dm_xy in the class doc string.
    Kxy_inv : DenMat
        inverse of Kxy, stored here for multiple re-use
    Kxy_log : DenMat
        log of Kxy, stored here for multiple re-use
    Kxy_proj0: DenMat
        Hermtian projection operator, projects zero eigenvalues of Kxy
    Kxy_proj1: DenMat
        Hermtian projection operator, projects non-zero eigenvalues of Kxy
    den_mat : DenMat
        input density matrix
    method : str
        method used to calculate logs and exponentials of matrices.
        Either 'exact-eigen', exact-pade', '2nd-order'
    num_ab_steps : int
        number of iterations of Arimoto Blahut algo
    num_bstrap_steps : int
        number of bootstrap steps. see DenMatPertTheory. This attribute is
        only used if method is '2nd-order'
    num_hidden_sts : int
        number of hidden states indexed by alpha, so this is number of alphas
    verbose : bool
    x_axes : list[int]
        This class calculates entanglement between x_axes and y_axes. x_axes
        and y_axes are mutually disjoint lists whose union is range(len(
        den_mat.row_shape))
    y_axes : list[int]

    """
    def __init__(self, den_mat, num_hidden_sts,
                 method, num_ab_steps, num_bstrap_steps=1, verbose=False):
        """
        Constructor

        Parameters
        ----------
        den_mat : DenMat
        num_hidden_sts : int
            number of values of h in dm_xyh defined in class doc string
        method : str
        num_ab_steps : int
        num_bstrap_steps : int
        verbose : bool

        Returns
        -------


        """
        Entang.__init__(self, len(den_mat.row_shape))
        self.den_mat = den_mat
        self.num_hidden_sts = num_hidden_sts
        self.method = method
        assert method in ['exact-eigen', 'exact-pade', '2nd-order']
        self.num_ab_steps = num_ab_steps
        self.num_bstrap_steps = num_bstrap_steps
        self.verbose = verbose

        self.x_axes = None
        self.y_axes = None
        self.Kxy = None
        self.Kxy_proj0 = None
        self.Kxy_proj1 = None
        self.Kxy_inv = None
        self.Kxy_log = None

    def exp(self, dm):
        """
        This method is a simple wrapper for dm.exp() so that all usages
        inside the class utilize the same method self.method which is global
        to the class.

        Parameters
        ----------
        dm : DenMat

        Returns
        -------
        DenMat

        """
        dm_out = None
        if self.method == 'exact-eigen':
            dm_out = dm.exp('eigen')
        elif self.method == 'exact-pade':
            dm_out = dm.exp('pade')
        elif self.method == '2nd-order':
            esys = DenMatPertTheory.get_bstrap_fin_eigen_sys_m(
                dm, self.num_bstrap_steps)
            dm_out = DenMatPertTheory.get_fun_of_dm_to_2nd_order(
                dm.num_rows, dm.row_shape, esys, np.exp)
        else:
            assert False
        return dm_out

    def log(self, dm, clipped=True):
        """
        This method is a simple wrapper for dm.log() so that all usages
        inside the class utilize the same method self.method which is global
        to the class.
        
        Parameters
        ----------
        dm : DenMat
        clipped: bool
             clips logs (see ut.clipped_log_of_vec) iff this is True
        Returns
        -------
        DenMat

        """
        dm_out = None
        if self.method == 'exact-eigen':
            dm_out = dm.log('eigen', clipped)
        elif self.method == 'exact-pade':
            dm_out = dm.log('pade', clipped)
        elif self.method == '2nd-order':
            esys = DenMatPertTheory.get_bstrap_fin_eigen_sys_m(
                dm, self.num_bstrap_steps)
            fun = np.log
            if clipped:
                fun = ut.clipped_log_of_vec
            dm_out = DenMatPertTheory.get_fun_of_dm_to_2nd_order(
                dm.num_rows, dm.row_shape, esys, fun)
        else:
            assert False
        return dm_out

    def get_entang(self, traced_axes_set):
        """
        This method returns the squashed entanglement E_xy, where x =
        traced_axes_set, and y is the set of all other axes.

        Parameters
        ----------
        traced_axes_set : set[int]

        Returns
        -------
        float

        """
        self.x_axes = list(traced_axes_set)
        num_row_axes = len(self.den_mat.row_shape)
        self.y_axes = [k for k in range(num_row_axes) if k not in self.x_axes]
        self.Kxy = self.den_mat.get_Kxy(self.x_axes, self.y_axes)
        self.Kxy_proj0, self.Kxy_proj1 = self.Kxy.get_eigenvalue_proj_ops()

        self.Kxy_inv = self.Kxy.pseudo_inv()
        # print("Kxy\n", self.Kxy)
        # print("Kxy_inv\n", self.Kxy_inv)
        self.Kxy_log = self.log(self.Kxy)
        Kxy_a = []

        # initial Kxy_a
        w0_a = np.random.uniform(size=self.num_hidden_sts)
        w0_a /= w0_a.sum()
        if self.verbose:
            print('\nw0_a=', w0_a)
        for alp in range(self.num_hidden_sts):
            Kxy_a_alp = DenMat(self.Kxy.num_rows, self.Kxy.row_shape)
            Kxy_a_alp.set_arr_to_zero()
            new_diag = np.random.uniform(size=self.Kxy.num_rows)
            new_diag *= w0_a[alp]/new_diag.sum()
            Kxy_a_alp.replace_diag_of_arr(new_diag)
            Kxy_a.append(Kxy_a_alp)

        entang = 0
        for time in range(self.num_ab_steps):
            entang, Kxy_a = self.next_step(Kxy_a)
            if self.verbose:
                print('------------time=', time)
                w_a = np.array([
                    None if Kxy_a[alp] is None else Kxy_a[alp].trace()
                    for alp in range(self.num_hidden_sts)])
                print("w_a=", w_a)
                # print('dist(w0_a, w_a)=', np.linalg.norm(w0_a-w_a),
                #       'sum w_a=', np.sum(w_a))
                sum_a_Kxy_a = DenMat.constant_dm_like(self.Kxy, 0j)
                for alp in range(self.num_hidden_sts):
                    print('Kxy_a for a=', alp, '\n', Kxy_a[alp])
                    if Kxy_a[alp] is not None:
                        sum_a_Kxy_a += Kxy_a[alp]
                # print('sum_xya_Kxy_a=', sum_a_Kxy_a.trace())
                print('dist(sum_a_Kxy_a, Kxy)=',
                      np.linalg.norm((sum_a_Kxy_a - self.Kxy).arr))
                print('entang=', entang)
        sum_a_Kxy_a = DenMat.constant_dm_like(self.Kxy, 0j)
        for x in Kxy_a:
            if x is not None:
                sum_a_Kxy_a += x
        print('\nfinal dist(sum_a_Kxy_a, Kxy)=',
              np.linalg.norm((sum_a_Kxy_a - self.Kxy).arr))
        print('final entang=', entang)
        return entang

    def next_step(self, Kxy_a):
        """
        This method is used self.num_ab_steps times, internally, in the
        method self.get_entang(). Given the current Kxy_a as input,
        it returns the next estimate of E_xy and the next Kxy_a.


        Parameters
        ----------
        Kxy_a : list[DenMat]
             a list of unnormalized density matrices Kxy_a[alp] such that

             trace(Kxy_a[alp]) = w_a[alp] for all alp
             sum_alp Kxy_a[alp] = Kxy,

             where list w_a is a probability distribution, and where alp
             ranges over range(self.num_hidden_sts)

        Returns
        -------
        float, list[DenMat]

        """
        num_x_axes = len(self.x_axes)
        num_y_axes = len(self.y_axes)
        set_x = set(range(num_x_axes))
        set_y = set(range(num_x_axes, num_x_axes + num_y_axes, 1))
        KDxy = DenMat(self.den_mat.num_rows, self.den_mat.row_shape)
        KDxy.set_arr_to_zero()

        # this loop will fill list log_Rxy_a
        # and KDxy
        log_Rxy_a = []
        for alp in range(self.num_hidden_sts):
            Kxy_a_alp = Kxy_a[alp]
            if Kxy_a_alp is None:
                log_Rxy_a_alp = None
            else:
                w_a_alp = Kxy_a_alp.trace()
                assert w_a_alp <= 1.
                Kx_a_alp = Kxy_a_alp.get_partial_tr(set_y)
                Ky_a_alp = Kxy_a_alp.get_partial_tr(set_x)
                if w_a_alp < 1e-6:
                    log_Rxy_a_alp = None
                else:
                    # Ix = DenMat.constant_dm_like(Kx_a_alp, 1)
                    # Iy = DenMat.constant_dm_like(Ky_a_alp, 1)
                    # # print('..,. evas Kx_a_alp',
                    # #       np.linalg.eigvalsh(Kxy_a_alp.arr))
                    # log_Rxy_a_alp = \
                    #     DenMat.get_kron_prod_of_den_mats(
                    #     [self.log(Kx_a_alp), Iy]) \
                    #     + DenMat.get_kron_prod_of_den_mats(
                    #     [Ix, self.log(Ky_a_alp*(1/w_a_alp))])
                    log_Rxy_a_alp = self.log(
                        DenMat.get_kron_prod_of_den_mats(
                            [Kx_a_alp, Ky_a_alp*(1/w_a_alp)]))
                    # print('log_Rxy_a_alp\n', log_Rxy_a_alp)
                    KDxy += Kxy_a_alp*(self.log(Kxy_a_alp)-log_Rxy_a_alp)
            log_Rxy_a.append(log_Rxy_a_alp)
        entang = KDxy.trace()/2

        # this loop will fill list new_Kxy_a
        new_Kxy_a = []
        sum_xya_Kxy_a = 0
        for alp in range(self.num_hidden_sts):
            Kxy_a_alp = Kxy_a[alp]
            if Kxy_a_alp is None:
                w_a_alp = None
            else:
                w_a_alp = Kxy_a_alp.trace()

            # print('======', alp, w_a_alp)
            if log_Rxy_a[alp] is None:
                new_Kxy_a.append(None)
            else:
                arg_of_exp = log_Rxy_a[alp] + KDxy
                arg_of_exp = self.Kxy_inv*arg_of_exp*self.Kxy_proj1
                # print('arg_of_exp\n', arg_of_exp)
                new_Kxy_a_alp = self.exp(arg_of_exp) - self.Kxy_proj0
                # print('--********', alp, arg_of_exp )
                new_Kxy_a.append(new_Kxy_a_alp)
                sum_xya_Kxy_a += new_Kxy_a_alp.trace()
        print('sum_xya_Kxy_a=', sum_xya_Kxy_a)
        new_Kxy_a = [None if dm is None else dm*(1/sum_xya_Kxy_a)
                     for dm in new_Kxy_a]
        return entang, new_Kxy_a

if __name__ == "__main__":
    def main1():
        print('-------------------------------main1')
        np.random.seed(123)
        dm = DenMat(8, (2, 2, 2))
        evas_of_dm_list = [
            np.array([.07, .03, .25, .15, .3, .1, .06, .04])
            , np.array([.05, .05, .2, .2, .3, .1, .06, .04])
            ]
        num_hidden_sts = 100
        num_ab_steps = 5
        print('num_hidden_sts=', num_hidden_sts)
        print('num_ab_steps=', num_ab_steps)
        for evas_of_dm in evas_of_dm_list:
            evas_of_dm /= np.sum(evas_of_dm)
            print('***************new dm')
            print('evas_of_dm\n', evas_of_dm)
            dm.set_arr_to_rand_den_mat(evas_of_dm)
            for method in ['exact-eigen']:
                print('-----method=', method)
                squasho = SquashedEnts(dm, num_hidden_sts,
                                       method, num_ab_steps, verbose=False)
                print('ent_02_1=', squasho.get_entang({0, 2}))

    from SymNupState import *

    def main2():
        print('-------------------------------main2')
        num_bits = 4
        num_up = 1
        dm1 = DenMat(1 << num_bits, tuple([2]*num_bits))
        st = SymNupState(num_up, num_bits)
        st_vec = st.get_st_vec()
        dm1.set_arr_from_st_vec(st_vec)
        num_hidden_sts = 4
        num_ab_steps = 5
        print('num_hidden_sts=', num_hidden_sts)
        print('num_ab_steps=', num_ab_steps)
        for method in ['exact-eigen']:
            print('-----method=', method)
            squasho = SquashedEnts(
                dm1, num_hidden_sts, method, num_ab_steps, verbose=False)
            print('entang_023: calculated value, known value\n',
                  squasho.get_entang({0, 2, 3}),
                  st.get_entang(3))
            print('entang_02: calculated value, known value\n',
                  squasho.get_entang({0, 2}),
                  st.get_entang(2))
            print('entang_1: calculated value, known value\n',
                  squasho.get_entang({1}),
                  st.get_entang(1))


    main1()
    main2()

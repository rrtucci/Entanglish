from entanglish.EntangCase import *
from entanglish.MaxEntangState import *
from entanglish.PureStEnt import *
import numpy as np


class SquashedEnt(EntangCase):
    """
    This class is a child of class EntangCase. Its purpose is to calculate
    the (bipartite) quantum entanglement Exy of a mixed state density matrix
    Dxy with parts x and y. Exy is defined here as the squashed
    entanglement. (See Wikipedia Ref. 1 for more detailed description and
    original references of squashed entanglement.)

    Exy = (1/2)*min S(x : y | h)

    where S(x : y | h) is the conditional mutual information (CMI,
    pronounced "see me") for a density matrix Dxyh. The minimum is over all
    Dxyh such that Dxy = trace_h Dxyh, where Dxy is a given, fixed density
    matrix.

    The squashed entanglement reduces to (1/2)*[S(x) + S(y)] = S(x) = S(y)
    when Dxy is a pure state. This is precisely the definition of
    entanglement for pure states that we use in the class PureStEnt.

    To calculate squashed entanglement, this class uses an algo which is a
    generalization of the Arimoto Blahut (AB) algorithm of classical
    information theory, where it is used to calculate channel capacities.

    See Ref.2 for a more detailed explanation of the algos used in this class.

    References
    ----------
    1. `<https://en.wikipedia.org/wiki/Squashed_entanglement>`_

    2. R.R. Tucci, "A New  Algorithm for Calculating Squashed Entanglement
    and a Python Implementation Thereof"

    Attributes
    ----------
    Dxy : DenMat          
        The input density matrix den_mat, after permuting its axes so that 
        the axes are given in the precise order self.x_axes + self.y_axes.
    den_mat : DenMat
    calc_formation_ent : bool
        True iff want to calculate entanglement of formation
    eps_w : float
        weights (probabilities) < eps_w are rounded to zero
    eps_log : float | None
        logs larger than log(eps_log) are clipped to log(eps_log)
    num_ab_steps : int
        number of iterations of Arimoto Blahut algo
    num_hidden_states : int
        number of hidden states indexed by alpha, so this is number of
        alphas. In the above class docstring, alpha is denoted by h,
        for hidden. This number is <= Dxy.num_rows^2.
    recursion_init : str
        How to initiate the recursion of Kxy_a.
    x_axes : list[int]
        This class calculates entanglement for 2 parts: x_axes and y_axes.
        x_axes and y_axes are mutually disjoint lists whose union is range(
        len( den_mat.row_shape))
    y_axes : list[int]

    """
    def __init__(self, den_mat, num_ab_steps, num_hidden_states=0,
                approx="eigen", recursion_init='eigen',
                verbose=False):
        """
        Constructor

        Parameters
        ----------
        den_mat : DenMat
        num_hidden_states : int
            number of states of h in the Dxyh defined in class doc string
        num_ab_steps : int
        approx : str
        recursion_init : str
        verbose : bool

        Returns
        -------

        """
        assert approx != "pert", "evaluation of squashed ent not implemented' \
                                 'for approx=='pert'"
        EntangCase.__init__(self, len(den_mat.row_shape), approx=approx,
                            verbose=verbose)
        self.den_mat = den_mat
        self.num_ab_steps = num_ab_steps
        self.num_hidden_states = num_hidden_states
        self.recursion_init = recursion_init

        self.eps_w = 1e-6
        self.eps_log = 1e-10

        self.x_axes = None
        self.y_axes = None
        self.Dxy = None
        self.Dxy_sqrt = None

        self.calc_formation_ent = False

    def regulate1(self, Kxy_a):
        """
        This internal method returns a list new_Kxy_a which is constructed
        by replacing each item Kxy_alp of list Kxy_a by its positive_part(),
        and then dividing the result by sum_alp tr_xy new_Kxy_a[alp]

        Some items of list Kxy_a may be None (those with very small w_alp).
        None items are not changed, they are left as None

        Parameters
        ----------
        Kxy_a : list[DenMat|None]

        Returns
        -------
        list[DenMat|None]

        """

        sum_xya_Kxy_a = 0
        new_Kxy_a = []
        for alp in range(len(Kxy_a)):
            Kxy_alp = Kxy_a[alp]
            if Kxy_alp is not None:
                Kxy_alp = Kxy_alp.positive_part()
                sum_xya_Kxy_a += Kxy_alp.trace()
            new_Kxy_a.append(Kxy_alp)

        new_Kxy_a = [(Kxy_alp * (1 / sum_xya_Kxy_a) if Kxy_alp is not None
                     else None) for Kxy_alp in new_Kxy_a]
        return new_Kxy_a

    def regulate2(self, Kxy_a):
        """
        This internal method returns a list new_Kxy_a which is constructed
        by replacing each item Kxy_alp of list Kxy_a by root*Kxy_alp*root^H,
        where root = sqrt(Dxy)* (sum_alp Kxy_alp)^(-1/2).

        Some items of list Kxy_a may be None (those with very small w_alp).
        None items are not changed, they are left as None

        Parameters
        ----------
        Kxy_a : list[DenMat|None]

        Returns
        -------
        list[DenMat|None]

        """
        sumk = DenMat(self.Dxy.num_rows, self.Dxy.row_shape)
        sumk.set_arr_to_zero()
        for alp in range(len(Kxy_a)):
            Kxy_alp = Kxy_a[alp]
            if Kxy_alp is not None:
                sumk += Kxy_alp
        evas, evec_cols = np.linalg.eigh(sumk.arr)
        evas = np.array([1/np.sqrt(x) if x > 1e-8 else 0 for x in evas])
        sqrt_inv_sumk = \
            DenMat(self.Dxy.num_rows, self.Dxy.row_shape,
                   arr=ut.herm_arr_from_eigen_sys(evas, evec_cols))
        root = self.Dxy_sqrt*sqrt_inv_sumk
        root_h = root.herm()
        new_Kxy_a = [(root*Kxy_alp*root_h if Kxy_alp is not None
                     else None) for Kxy_alp in Kxy_a]
        return new_Kxy_a

    def get_entang(self, axes_subset):
        """
        This method returns the squashed entanglement Exy, where x =
        axes_subset, and y is the set of all other axes.

        Parameters
        ----------
        axes_subset : set[int]

        Returns
        -------
        float

        """
        self.x_axes = list(axes_subset)
        num_x_axes = len(self.x_axes)
        num_row_axes = len(self.den_mat.row_shape)
        self.y_axes = [k for k in range(num_row_axes) if k not in self.x_axes]
        num_y_axes = len(self.y_axes)
        self.Dxy = self.den_mat.get_rho_xy(self.x_axes, self.y_axes)
        self.Dxy_sqrt = self.sqrt(self.Dxy)

        Kxy_a = []
        if self.recursion_init == 'stat-pt':
            # this turns out to be a stationary point of the recursion
            # relation
            assert self.num_hidden_states != 0,\
                'num_hidden_states must be specified'
            w_alp = 1 / self.num_hidden_states
            Kxy_alp = self.Dxy*w_alp
            Kxy_a = [Kxy_alp]*self.num_hidden_states
        elif self.recursion_init in ['eigen', 'eigen+']:
            if self.recursion_init[-1] != '+':
                self.num_hidden_states = self.Dxy.num_rows
            else:
                self.num_hidden_states = self.Dxy.num_rows**2
            evas, evec_cols = np.linalg.eigh(self.Dxy.arr)
            if self.recursion_init[-1] != '+':
                for alp in range(self.num_hidden_states):
                    Kxy_alp = DenMat(self.Dxy.num_rows, self.Dxy.row_shape)
                    eva = evas[alp]
                    if eva < self.eps_w:
                        Kxy_alp = None
                    else:
                        Kxy_alp.set_arr_from_st_vec(evec_cols[:, alp])
                        Kxy_alp = Kxy_alp*eva
                    Kxy_a.append(Kxy_alp)
            else:
                # this gives indices of evas in decreasing order
                indices = np.flip(np.argsort(evas))

                evas = np.array([evas[k] for k in indices])
                evec_cols = np.stack(
                    [evec_cols[:, k] for k in indices], axis=1)

                num_nonzero_evas = self.Dxy.num_rows
                for k in range(len(evas)):
                    if evas[k] < self.eps_w:
                        num_nonzero_evas = k
                        break
                if num_nonzero_evas > 1:
                    eps = evas[num_nonzero_evas-1]/(2*(num_nonzero_evas-1))
                else:
                    eps = 0
                eps9 = np.sqrt(eps*(1-eps))
                z = (1 + 1j)/np.sqrt(2)
                zc = np.conj(z)
                diag_ind = [(x, x) for x in range(self.Dxy.num_rows)]
                non_diag_ind = [(row, col)
                                for row in range(self.Dxy.num_rows)
                                for col in range(self.Dxy.num_rows)
                                if row != col]
                indices = diag_ind + non_diag_ind
                assert len(indices) == self.Dxy.num_rows**2
                alp = -1
                for row, col in indices:
                    alp += 1
                    if row == col:
                        if row < num_nonzero_evas:
                            if num_nonzero_evas > 1:
                                w_alp = evas[alp] - (num_nonzero_evas-1)*eps
                            else:
                                w_alp = evas[alp]
                            assert w_alp > 0
                        else:
                            w_alp = 0.0
                    elif row > num_nonzero_evas-1 or col > num_nonzero_evas-1:
                        w_alp = 0.0
                    else:
                        w_alp = eps
                    if w_alp < self.eps_w:
                        Kxy_a.append(None)
                        continue
                    Kxy_alp = DenMat(self.Dxy.num_rows, self.Dxy.row_shape)
                    Kxy_alp.set_arr_to_zero()
                    if alp < num_nonzero_evas:
                        Kxy_alp[alp, alp] = w_alp
                    elif alp >= self.Dxy.num_rows:
                        Kxy_alp[row, row] = (1 - eps) * w_alp
                        Kxy_alp[col, col] = eps * w_alp
                        if row < col:
                            Kxy_alp[row, col] = z*eps9*w_alp
                            Kxy_alp[col, row] = zc*eps9*w_alp
                        else:
                            Kxy_alp[row, col] = -zc*eps9*w_alp
                            Kxy_alp[col, row] = -z*eps9*w_alp
                    else:
                        # should never reach here because of continue
                        assert False
                    Kxy_alp.arr = ut.switch_arr_basis(Kxy_alp.arr, evec_cols,
                                                      reverse=True)
                    Kxy_a.append(Kxy_alp)

        else:
            assert False, 'unexpected recursion_init'
        if self.verbose:
            print("\ninitial norm of Dxy - sum_alp Kxy_alp, should be 0\n",
                np.linalg.norm(self.Dxy.arr -
                sum([Kxy_alp.arr for Kxy_alp in Kxy_a
                     if Kxy_alp is not None])))

        entang = -1
        for step in range(self.num_ab_steps):
            Kxy_a, entang, err = self.next_step(Kxy_a)
            if self.verbose:
                print('--ab step=', step,
                      ', entang=', "{:.6f}".format(entang),
                      ", err=", "{:.6f}".format(err))
        return entang

    def next_step(self, Kxy_a):
        """
        This method is used self.num_ab_steps times, internally, in the
        method self.get_entang(). Given the current Kxy_a as input,
        it returns the next estimate of Kxy_a , Exy , err

        Kxy_a is a list of un-normalized density matrices such that Kxy_a[
        alp]=Dxy_a[alp]*w_a[alp] and sum_alp Kxy_a[alp] = Dxy.

        Exy is the entanglement for parts x and y.

        If

        Delta_xy[alp]= log Dxy_a[alp] - log(Dx_a[alp]Dy_a[alp]),

        then err is a float that measures the variance in the
        Delta_xy[alp] matrices. This variance tends to zero as num_ab_steps
        tends to infinity

        mean_alp x[alp] = weighted average (with weights w_a) over alp of x[
        alp]

        err = mean_alp norm(Delta_xy[alp] - mean_alp Delta_xy[alp])

        Parameters
        ----------
        Kxy_a : list[DenMat]

        Returns
        -------
        list[DenMat], float, float

        """
        num_x_axes = len(self.x_axes)
        num_y_axes = len(self.y_axes)
        set_x = set(range(num_x_axes))
        set_y = set(range(num_x_axes, num_x_axes + num_y_axes, 1))

        log_Dxy_a = []
        log_Dx_Dy_a = []
        w_a = []
        Delta_xy = DenMat(self.Dxy.num_rows, self.Dxy.row_shape)
        Delta_xy.set_arr_to_zero()
        num_nonzero_w_alp = 0
        entang = 0
        for alp in range(self.num_hidden_states):
            Kxy_alp = Kxy_a[alp]
            if Kxy_alp is None:
                w_alp = 0.0
            else:
                w_alp = Kxy_alp.trace()
                if w_alp < self.eps_w:
                    w_alp = 0
                    Kxy_alp = None
                else:
                    num_nonzero_w_alp += 1
            w_a.append(w_alp)
            if Kxy_alp is None:
                log_Dxy_alp = None
                log_Dx_Dy_alp = None
            else:
                Dxy_alp = Kxy_alp*(1/w_alp)
                log_Dxy_alp = self.log(Dxy_alp, eps=self.eps_log)

                Dx_alp = Dxy_alp.get_partial_tr(set_y)
                Dy_alp = Dxy_alp.get_partial_tr(set_x)
                log_Dx_Dy_alp = self.log(DenMat.get_kron_prod_of_den_mats(
                                [Dx_alp, Dy_alp]), eps=self.eps_log)

                Delta_xy_alp = log_Dxy_alp - log_Dx_Dy_alp
                Delta_xy += Delta_xy_alp*w_alp
                # print('cccvvvbbb', Dxy_alp,  w_alp)
                if not self.calc_formation_ent:
                    entang += (Dxy_alp * Delta_xy_alp).trace() * w_alp / 2
                else:  # more precise
                    ecase = PureStEnt(Dxy_alp, approx=self.approx)
                    entang += ecase.get_entang(set_x)*w_alp
            log_Dxy_a.append(log_Dxy_alp)
            log_Dx_Dy_a.append(log_Dx_Dy_alp)

        err = 0
        for alp in range(self.num_hidden_states):
            if log_Dxy_a[alp] is not None:
                Delta_xy_alp = log_Dxy_a[alp] - log_Dx_Dy_a[alp]
                err += (Delta_xy_alp - Delta_xy).norm()*w_a[alp]

        print_w_a = False
        if print_w_a:
            print('w_a=', w_a, 'sum=', np.sum(np.array(w_a)))

        new_Kxy_a = []
        for alp in range(self.num_hidden_states):
            if log_Dx_Dy_a[alp] is None:
                new_Kxy_alp = None
            else:
                arg_of_exp = Delta_xy + log_Dx_Dy_a[alp]
                evas, evec_cols = np.linalg.eigh(arg_of_exp.arr)
                if not self.calc_formation_ent:
                    # set positive evas to zero
                    evas = np.array([min(x, 0) for x in evas])
                    arr = ut.fun_of_herm_arr_from_eigen_sys(
                        np.exp, evas, evec_cols)
                    Dxy_alp = DenMat(
                        self.Dxy.num_rows, self.Dxy.row_shape, arr)
                    new_Kxy_alp = Dxy_alp*w_a[alp]
                else:
                    max_pos = np.argmax(evas)
                    vec = evec_cols[:, max_pos]
                    arr = np.outer(vec, np.conj(vec)) * np.exp(evas[max_pos])
                    new_Kxy_alp = DenMat(
                        self.Dxy.num_rows, self.Dxy.row_shape, arr)

            new_Kxy_a.append(new_Kxy_alp)

        new_Kxy_a = self.regulate2(new_Kxy_a)

        return new_Kxy_a, entang, err


if __name__ == "__main__":
    def main1():
        print('###############################main1')
        np.random.seed(123)
        dm = DenMat(8, (2, 2, 2))
        evas_of_dm_list = [
            np.array([.07, .03, .25, .15, .3, .1, .06, .04])
            , np.array([.05, .05, .2, .2, .3, .1, .06, .04])
            ]

        recursion_init = 'eigen+'
        num_ab_steps = 100
        print('recursion_init=', recursion_init)
        print('num_ab_steps=', num_ab_steps)
        for evas_of_dm in evas_of_dm_list:
            evas_of_dm /= np.sum(evas_of_dm)
            print('***************new dm')
            print('evas_of_dm\n', evas_of_dm)
            dm.set_arr_to_rand_den_mat(evas_of_dm)
            ecase = SquashedEnt(dm, num_ab_steps,
                recursion_init=recursion_init, verbose=True)
            print('ent_02_1=', ecase.get_entang({0, 2}))

    from entanglish.SymNupState import *

    def main2():
        print('###############################main2')
        num_qbits = 4
        num_up = 1
        dm1 = DenMat(1 << num_qbits, tuple([2]*num_qbits))
        st = SymNupState(num_up, num_qbits)
        st_vec = st.get_st_vec()
        dm1.set_arr_from_st_vec(st_vec)

        recursion_init = 'eigen+'
        num_ab_steps = 5
        print('recursion_init=', recursion_init)
        print('num_ab_steps=', num_ab_steps)
        ecase = SquashedEnt(dm1, num_ab_steps,
            recursion_init=recursion_init, verbose=True)
        print('entang_023: algo value, known value\n',
              ecase.get_entang({0, 2, 3}),
              st.get_known_entang(3))
        print('entang_02: algo value, known value\n',
              ecase.get_entang({0, 2}),
              st.get_known_entang(2))
        print('entang_1: algo value, known value\n',
              ecase.get_entang({1}),
              st.get_known_entang(1))

    main1()
    main2()

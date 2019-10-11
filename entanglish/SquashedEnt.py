from entanglish.EntangCase import *
from entanglish.MaxEntangState import *
from operator import itemgetter
import numpy as np
import random


class SquashedEnt(EntangCase):
    """
    This class is a child of class EntangCase. It's purpose is to calculate
    the (bipartite) quantum entanglement Exy of a mixed state density matrix
    Dxy with parts x and y. Exy is defined here as the squashed entanglement
    ( see Wikipedia Ref. 1 for more detailed description and original
    references of squashed entanglement)

    Exy = (1/2)*min[S(x : y | h)]

    where S(x : y | h) is the conditional mutual information (CMI,
    pronounced "see me") for a density matrix Dxyh. The minimum is over all
    Dxyh such that Dxy = trace_h Dxyh, where Dxy is given and fixed.

    The squashed entanglement reduces to (1/2)*[S(x) + S(y)] = S(x) = S(y)
    when Dxy is a pure state. This is precisely the definition of
    entanglement for pure states that we use in the class PureStEnt.

    To calculate squashed entanglement, this class uses an algo which is a
    generalization of the Arimoto Blahut (AB) algorithm of classical
    information theory, where it is used to calculate channel capacities.

    See Ref.2 for a more detailed explanation of the algos used in this class.

    References
    ----------
    1. https://en.wikipedia.org/wiki/Squashed_entanglement

    2. R.R. Tucci, "A New  Algorithm for Calculating Squashed Entanglement
    and a Python Implementation Thereof"

    Attributes
    ----------
    Dxy : DenMat          
        The input density matrix den_mat, after permuting its axes so that 
        the axes are given in the precise order self.x_axes + self.y_axes. 
    den_mat : DenMat
    num_ab_steps : int
        number of iterations of Arimoto Blahut algo
    num_hidden_states : int
        number of hidden states indexed by alpha, so this is number of alphas.
        In the above class docstring, alpha is denoted by h, for hidden.
    x_axes : list[int]
        This class calculates entanglement for 2 parts: x_axes and y_axes.
        x_axes and y_axes are mutually disjoint lists whose union is range(
        len( den_mat.row_shape))
    y_axes : list[int]

    """
    def __init__(self, den_mat, num_hidden_states,
                 num_ab_steps, method="eigen", verbose=False):
        """
        Constructor

        Parameters
        ----------
        den_mat : DenMat
        num_hidden_states : int
            number of states of h in the Dxyh defined in class doc string
        num_ab_steps : int
        method : str
        verbose : bool

        Returns
        -------


        """
        assert method != "pert", "evaluation of squashed ent not implemented' \
                                 'for method=='pert'"
        EntangCase.__init__(self, len(den_mat.row_shape), method=method,
                            verbose=verbose)
        self.den_mat = den_mat
        self.num_hidden_states = num_hidden_states
        self.num_ab_steps = num_ab_steps

        self.x_axes = None
        self.y_axes = None
        self.Dxy = None

    @staticmethod
    def regulate(Kxy_a):
        """
        This internal method returns a list new_Kxy_a which constructed by
        replacing each item Kxy_alp of list Kxy_a by its positive part
        normalized so that sum_xya of all items of new_Kxy_a is one.


        Parameters
        ----------
        Kxy_a : list[DenMat]

        Returns
        -------
        list[DenMat]

        """

        sum_xya_Kxy_a = 0
        new_Kxy_a = []
        for alp in range(len(Kxy_a)):
            Kxy_alp = Kxy_a[alp].positive_part()
            sum_xya_Kxy_a += Kxy_alp.trace()
            new_Kxy_a.append(Kxy_alp)
        new_Kxy_a = [Kxy_alp * (1 / sum_xya_Kxy_a) for Kxy_alp in new_Kxy_a]
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

        # w_alp = 1 / self.num_hidden_states
        # # initial Kxy_a
        # # Kxy_a[alp] = Dxy/num_hidden_steps,
        # # for  all alp
        # Kxy_alp = self.Dxy*w_alp
        # Kxy_a = [Kxy_alp]*self.num_hidden_states

        w_a = np.random.rand(self.num_hidden_states)
        w_a /= np.sum(w_a)
        dd = np.diag(self.Dxy.arr)
        Kxy_a = []
        for alp in range(self.num_hidden_states):
            if alp == 0:
                Kxy_alp = self.Dxy.copy()
            else:
                Kxy_alp = DenMat(self.Dxy.num_rows, self.Dxy.row_shape)
                Kxy_alp.set_arr_to_zero()
            Kxy_alp.replace_diag_of_arr(dd * (1/self.num_hidden_states))
            Kxy_a.append(Kxy_alp)

        Kxy_a = SquashedEnt.regulate(Kxy_a)

        entang = 0
        print('')
        for step in range(self.num_ab_steps):
            Kxy_a, entang = self.next_step(Kxy_a)
            if self.verbose:
                print('----------ab step=', step, 'entang=', entang)
        return entang

    def next_step(self, Kxy_a):
        """
        This method is used self.num_ab_steps times, internally, in the
        method self.get_entang(). Given the current Kxy_a as input,
        it returns the next estimate of Kxy_a and Exy.


        Parameters
        ----------
        Kxy_a : list[DenMat]
            a list of un-normalized density matrices

            Kxy_a[alp]=Dxy_a[alp]*w_a[alp].

            Therefore,

            sum_alp Kxy_a[alp] = Dxy,

        Returns
        -------
        list[DenMat], float

        """
        num_x_axes = len(self.x_axes)
        num_y_axes = len(self.y_axes)
        set_x = set(range(num_x_axes))
        set_y = set(range(num_x_axes, num_x_axes + num_y_axes, 1))

        log_Dx_Dy_a = []
        w_a = []
        alp_rand = random.randrange(self.num_hidden_states)
        for alp in range(self.num_hidden_states):
            Kxy_alp = Kxy_a[alp]
            w_alp = Kxy_alp.trace()
            w_a.append(w_alp)
            Dxy_alp = Kxy_alp*(1/w_alp)
            log_Dxy_alp = self.log(Dxy_alp)

            Dx_alp = Dxy_alp.get_partial_tr(set_y)
            Dy_alp = Dxy_alp.get_partial_tr(set_x)
            log_Dx_Dy_alp = self.log(DenMat.get_kron_prod_of_den_mats(
                            [Dx_alp, Dy_alp]))
            log_Dx_Dy_a.append(log_Dx_Dy_alp)

            if alp == alp_rand:
                lam_xy = log_Dxy_alp - log_Dx_Dy_alp
        if self.verbose:
            print('w_a=', w_a, 'sum=', np.sum(np.array(w_a)))

        new_Kxy_a = []
        sum_numerator = DenMat(self.Dxy.num_rows, self.Dxy.row_shape)
        sum_numerator.set_arr_to_zero()
        for alp in range(self.num_hidden_states):
            new_Kxy_alp = self.exp(lam_xy + log_Dx_Dy_a[alp])*w_a[alp]
            new_Kxy_a.append(new_Kxy_alp)
            sum_numerator += new_Kxy_alp
        inv_sum_numerator = sum_numerator.inv()
        for alp in range(self.num_hidden_states):
            # print('rrrrr', np.linalg.norm(self.Dxy.arr),
            #             np.linalg.norm(inv_sum_numerator.arr),
            #             np.linalg.norm(new_Kxy_a[alp].arr))
            new_Kxy_alp = self.Dxy*inv_sum_numerator*new_Kxy_a[alp]
            new_Kxy_alp = (new_Kxy_alp + new_Kxy_alp.herm())*(1/2)
            # print('rrrrr111', np.linalg.norm(new_Kxy_alp.arr))
            new_Kxy_a[alp] = new_Kxy_alp

        new_Kxy_a = SquashedEnt.regulate(new_Kxy_a)
        # print('nnnnnnnnnnnnn', np.sum(np.array([x.trace() for x in
        #                                         new_Kxy_a])))
        entang = (self.Dxy*lam_xy).trace()/2

        return new_Kxy_a, entang


if __name__ == "__main__":
    def main1():
        print('###############################main1')
        np.random.seed(123)
        dm = DenMat(8, (2, 2, 2))
        evas_of_dm_list = [
            np.array([.07, .03, .25, .15, .3, .1, .06, .04])
            , np.array([.05, .05, .2, .2, .3, .1, .06, .04])
            ]
        num_hidden_states = 10
        num_ab_steps = 10
        print('num_hidden_states=', num_hidden_states)
        print('num_ab_steps=', num_ab_steps)
        for evas_of_dm in evas_of_dm_list:
            evas_of_dm /= np.sum(evas_of_dm)
            print('***************new dm')
            print('evas_of_dm\n', evas_of_dm)
            dm.set_arr_to_rand_den_mat(evas_of_dm)
            ecase = SquashedEnt(dm, num_hidden_states, num_ab_steps,
                                verbose=False)
            print('ent_02_1=', ecase.get_entang({0, 2}))

    from entanglish.SymNupState import *

    def main2():
        print('###############################main2')
        num_bits = 4
        num_up = 1
        dm1 = DenMat(1 << num_bits, tuple([2]*num_bits))
        st = SymNupState(num_up, num_bits)
        st_vec = st.get_st_vec()
        dm1.set_arr_from_st_vec(st_vec)

        num_hidden_states = 10
        num_ab_steps = 50
        print('num_hidden_states=', num_hidden_states)
        print('num_ab_steps=', num_ab_steps)
        ecase = SquashedEnt(
            dm1, num_hidden_states, num_ab_steps, verbose=False)
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

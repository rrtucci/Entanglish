from entanglish.EntangCase import *
from entanglish.MaxEntangState import *
from operator import itemgetter
import numpy as np


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
        Dxy_a = []

        # initial Dxy_a
        # Dxy_a[0] = Dxy,
        # all others are max entangled
        dm_max_ent = DenMat(self.Dxy.num_rows, self.Dxy.row_shape)
        x_axes0 = list(range(num_x_axes))
        y_axes0 = list(range(num_x_axes, num_row_axes, 1))
        max_ent_st = MaxEntangState(dm_max_ent.num_rows, dm_max_ent.row_shape,
                                    x_axes0, y_axes0)
        EntangCase.check_max_entang_st(max_ent_st)
        st_vec = max_ent_st.get_st_vec()
        entang = max_ent_st.get_known_entang()
        dm_max_ent.set_arr_from_st_vec(st_vec)
        # print('dddddd dm max ent', dm_max_ent.arr)
        for alp in range(self.num_hidden_states):
            if alp == 0:
                Dxy_alp = self.Dxy
            else:
                Dxy_alp = dm_max_ent
            Dxy_a.append(Dxy_alp)

        for step in range(self.num_ab_steps):
            if self.verbose:
                print('------------ab step=', step)
                print('entang=', entang)
            entang, Dxy_a = self.next_step(Dxy_a)
        if self.verbose:
            print('-----------\nfinal entang=', entang)
        return entang

    def next_step(self, Dxy_a):
        """
        This method is used self.num_ab_steps times, internally, in the
        method self.get_entang(). Given the current Dxy_a as input,
        it returns the next estimate of Exy and the next Dxy_a.


        Parameters
        ----------
        Dxy_a : list[DenMat]
             a list of normalized density matrices Dxy_a[alp] such that

             sum_alp w_alp * Dxy_a[alp] = Dxy,

        Returns
        -------
        float, list[DenMat]

        """
        num_x_axes = len(self.x_axes)
        num_y_axes = len(self.y_axes)
        set_x = set(range(num_x_axes))
        set_y = set(range(num_x_axes, num_x_axes + num_y_axes, 1))

        log_Dxy_a = []
        log_Dx_Dy_a = []
        entang_a = []
        for alp in range(self.num_hidden_states):
            Dxy_alp = Dxy_a[alp]
            log_Dxy_alp = self.log(Dxy_alp)
            log_Dxy_a.append(log_Dxy_alp)

            Dx_alp = Dxy_alp.get_partial_tr(set_y)
            Dy_alp = Dxy_alp.get_partial_tr(set_x)
            log_Dx_Dy_alp = self.log(DenMat.get_kron_prod_of_den_mats(
                            [Dx_alp, Dy_alp]))
            log_Dx_Dy_a.append(log_Dx_Dy_alp)

            entang_alp = (Dxy_alp*log_Dxy_alp -
                          Dxy_alp*log_Dx_Dy_alp).trace()/2
            if alp == -1:
                print("llll-norm", np.linalg.norm(Dxy_alp.arr),
                      Dxy_alp.trace())
                print('hhhhhhh', np.linalg.norm((log_Dxy_alp -
                                                 log_Dx_Dy_alp).arr))
                Dx_Dy_alp = DenMat.get_kron_prod_of_den_mats([Dx_alp, Dy_alp])
                print('vvvvvvvvv', np.linalg.norm(Dxy_alp.arr -
                    Dx_Dy_alp.arr))
                print('ccccccDx', Dx_alp)
                print('ccccccDy', Dy_alp)
                print('ccccccDxy', Dxy_alp)
                print('ccccccDxDy', Dx_Dy_alp)
                print(("bbbnnnmmm", (Dxy_alp*log_Dxy_alp).trace()/2,
                      -(Dxy_alp*log_Dx_Dy_alp).trace()/2))
            entang_a.append(entang_alp)
        alp_min, entang = min(enumerate(entang_a), key=itemgetter(1))
        if self.verbose:
            print('entang_a=', entang_a)
            print('alp_min=', alp_min)
        lam_xy = log_Dxy_a[alp_min] - log_Dx_Dy_a[alp_min]

        # this loop will fill list new_Dxy_a
        new_Dxy_a = []
        for alp in range(self.num_hidden_states):
            if alp == alp_min:
                new_Dxy_alp = self.Dxy
            else:
                new_Dxy_alp = self.exp(lam_xy + log_Dx_Dy_a[alp])
                new_Dxy_alp *= (1/new_Dxy_alp.trace())
            new_Dxy_a.append(new_Dxy_alp)
        # print('xxxxxxrrrrr 0=', np.linalg.norm(
        #     new_Dxy_a[alp_min].arr - self.Dxy.arr))
        return entang, new_Dxy_a


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
            ecase = SquashedEnt(dm, num_hidden_states, num_ab_steps)
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
        num_ab_steps = 40
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

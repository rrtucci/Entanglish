from entanglish.SquashedEnt import *


class FormationEnt(SquashedEnt):
    """
    This class is a child of class SquashedEnt. Its purpose is to calculate
    the (bipartite) quantum entanglement Exy of a mixed state density matrix
    Dxy with parts x and y. Exy is defined here as the entanglement of
    formation

    Exy = sum_alp w_a[alp]  min S(Dx_a[alp])

    where S(Dx_a[alp]) is the von Neumann entropy for  density matrix Dx_a[
    alp] = tr_y Dxy_a[alp]. The minimum is over all Dxy_a[alp] such that
    Dxy_a[alp] is a pure state |psi[alp]><psi[alp]|, and sum_alp w_a[ alp]
    Dxy_a[ alp] = Dxy where Dxy is a given, fixed density matrix.

    If we add to the definition of squashed entanglement the further
    constraint that Dxy_a[alp] is a pure state for all alp, then the
    squashed entanglement of Dxy becomes precisely the entanglement of
    formation of Dxy.

    In this class, most of the steps used for calculating entang of
    formation are the same as those for calculating squashed entang. Those
    steps that aren't are turned on or off with the bool flag do_formation_ent

    A closed exact formula is known, thanks to Wooters et al., for the
    entang of formation of an arbitrary mixture of 2 qubits. Class
    TwoQubitStates of entanglish contains an implementation of said formula.

    See Ref.1 for a detailed explanation of the algos used in this class.

    References
    ----------
    1. R.R. Tucci, "A New  Algorithm for Calculating Squashed Entanglement
    and a Python Implementation Thereof"


    """

    def __init__(self, *args, **kwargs):
        """
        Constructor

        Parameters
        ----------
        args :
            list of args of SquashedEnt constructor
        kwargs :
            dictionary of kwargs of SquashedEnt constructor
        """

        SquashedEnt.__init__(self, *args, **kwargs)
        self.do_formation_ent = True

if __name__ == "__main__":

    from entanglish.TwoQubitStates import *
    from entanglish.SymNupState import *

    def main1():
        print('###############################main1, rand, 3 qubits')
        np.random.seed(123)
        dm = DenMat(8, (2, 2, 2))
        evas_of_dm_list = [
            np.array([.07, .03, .25, .15, .3, .1, .06, .04])
            , np.array([.05, .05, .2, .2, .3, .1, .06, .04])
            ]
        num_hidden_states = 8
        num_ab_steps = 40
        print('num_hidden_states=', num_hidden_states)
        print('num_ab_steps=', num_ab_steps)
        for evas_of_dm in evas_of_dm_list:
            evas_of_dm /= np.sum(evas_of_dm)
            print('***************new dm')
            print('evas_of_dm\n', evas_of_dm)
            dm.set_arr_to_rand_den_mat(evas_of_dm)
            ecase = FormationEnt(dm, num_hidden_states, num_ab_steps,
                                recursion_init='equi-diag', verbose=True)
            print('ent_02_1=', ecase.get_entang({0, 2}))

    def main2():
        print('###############################main2, sym nup')
        num_bits = 4
        num_up = 1
        dm1 = DenMat(1 << num_bits, tuple([2]*num_bits))
        st = SymNupState(num_up, num_bits)
        st_vec = st.get_st_vec()
        dm1.set_arr_from_st_vec(st_vec)

        num_hidden_states = 16
        num_ab_steps = 5
        print('num_hidden_states=', num_hidden_states)
        print('num_ab_steps=', num_ab_steps)
        ecase = FormationEnt(dm1, num_hidden_states, num_ab_steps,
            recursion_init='eigen', verbose=True)
        print('entang_023: algo value, known value\n',
              ecase.get_entang({0, 2, 3}),
              st.get_known_entang(3))
        print('entang_02: algo value, known value\n',
              ecase.get_entang({0, 2}),
              st.get_known_entang(2))
        print('entang_1: algo value, known value\n',
              ecase.get_entang({1}),
              st.get_known_entang(1))

    def main3():
        print('###############################main3, werner 2 qubit')
        dm1 = TwoQubitStates.get_bell_basis_diag_dm(.7)
        num_hidden_states = 4
        num_ab_steps = 5
        print('num_hidden_states=', num_hidden_states)
        print('num_ab_steps=', num_ab_steps)
        for dm in [dm1]:
            print("-------new dm")
            formation_entang =\
                  TwoQubitStates.get_known_formation_entang(dm)

            ecase = FormationEnt(dm, num_hidden_states, num_ab_steps,
                recursion_init='equi-diag', verbose=True)
            print('entang_0: algo value, known value\n',
                  ecase.get_entang({1}), formation_entang)

    def main4():
        print('###############################main4, rand, 2 qubit')
        np.random.seed(123)
        dm2 = DenMat(4, (2, 2))
        dm2.set_arr_to_rand_den_mat(np.array([.1 , .2 , .3, .4]))
        dm3 = DenMat(4, (2, 2))
        dm3.set_arr_to_rand_den_mat(np.array([.1 , .1 , .1, .7]))
        num_hidden_states = 4
        num_ab_steps = 40
        print('num_hidden_states=', num_hidden_states)
        print('num_ab_steps=', num_ab_steps)
        for dm in [dm2, dm3]:
            print("-------new dm")
            formation_entang =\
                  TwoQubitStates.get_known_formation_entang(dm)

            ecase = FormationEnt(dm, num_hidden_states, num_ab_steps,
                recursion_init='eigen-sep', verbose=True)
            print('entang_0: algo value, known value\n',
                  ecase.get_entang({1}), formation_entang)

    main1()
    main2()
    main3()
    main4()

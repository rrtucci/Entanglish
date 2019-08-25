import itertools as it
import numpy as np
import entanglish.utilities as ut


class SymNupState:
    """
    This class is designed to perform tasks related to a SymNupState.
    SymNupState is an abbreviation for Symmetrized N-qubits-up State,
    which is a special, very convenient for testing purposes, type of
    quantum state vector. Note, this is a pure state of qubits only. No
    qudits with d != 2 in this state. The state contains a total of num_bits
    qubits. num_up of them are up (in state |1>) and num_bits - num_up are
    down (in state |0>). To build such a state, we first create any (
    normalized) initial state vector with the required number of up and down
    qubits, and then we apply a total symmetrizer to that initial state
    vector.

    It turns out that SymNupState's have a (bipartite) entanglement that is
    known and has a simple analytical expression given by the classical
    entropy of a hyper-geometric distribution.


    Attributes
    ----------
    num_bits : int
        total number of qubits in the state
    num_up : int
        should be <= num_bits. The number of qubits that is up (in state
        |1>). The other num_bits - n_up are down (in state |0>)

    """

    def __init__(self, num_up, num_bits):
        """
        Constructor

        Parameters
        ----------
        num_up : int
        num_bits : int

        Returns
        -------


        """
        assert 0 <= num_up <= num_bits
        self.num_bits = num_bits
        self.num_up = num_up

    def get_st_vec(self):
        """
        This method outputs the (pure) state vector for the SymNupState
        object.

        Returns
        -------
        np.ndarray

        """
        st_vec = np.zeros(tuple([2]*self.num_bits), dtype=complex)
        all_axes = list(range(0, self.num_bits))
        comb_len = self.num_up
        for up_axes in it.combinations(all_axes, comb_len):
            index = tuple([1 if k in up_axes else 0
                          for k in range(self.num_bits)])
            st_vec[index] = 1
        mag = np.linalg.norm(st_vec)
        st_vec /= mag
        return st_vec.reshape((1 << self.num_bits,))

    def get_entang(self, num_traced_axes):
        """
        This method calculates the (bipartite) entanglement

        E(x_axes, y_axes)=E(y_axes, x_axes) (order of x_axes and y_axes
        arguments doesn't matter)

        for the state, where len(x_axes)= num_traced_axes, and len(y_axes)=
        total_num_row_axes - num_traced_axes. Because of the symmetrization of
        the state, E(x_axes, y_axes) only depends of the lengths of x_axes
        and y_axes. One can prove that E(x_axes, y_axes) is expressible
        analytically as the classical entropy of a hyper-geometric
        distribution (
        https://en.wikipedia.org/wiki/Hypergeometric_distribution)

        Parameters
        ----------
        num_traced_axes : int

        Returns
        -------
        float

        """
        assert 0 <= num_traced_axes <= self.num_bits
        nn = self.num_bits
        n = self.num_bits - num_traced_axes
        xx = self.num_up
        probs = [ut.prob_hypergeometric(x, xx, n, nn)
                 for x in range(xx + 1)]
        return ut.get_entropy_from_probs(np.array(probs))

if __name__ == "__main__":
    def main():
        num_up = 4
        num_bits = 5
        st = SymNupState(num_up, num_bits)
        print('st_vec\n', st.get_st_vec())
        for num_traced_axes in range(0, num_bits+1):
            print('entang for ' + str(num_traced_axes) + ' traced axes=',
                  st.get_entang(num_traced_axes))
    main()

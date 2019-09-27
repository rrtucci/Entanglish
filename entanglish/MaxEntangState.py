import numpy as np


class MaxEntangState:
    """
    This class is designed to perform tasks related to a maximally entangled
    pure state with parts x_axes, y_axes. x_axes, y_axes give a bi-partition
    of range( len(row_shape)).

    See Ref.1 for an explicit definition of the maximally entangled states
    that we use. The basic requirement for a density matrix Dxy to be
    maximally entangled is for its partial trace Dx to be a diagonal matrix
    with all terms in the diagonal equal to the same constant. The sum of
    the diagonal elements must of course be one. For example, Dx=diag(0.25,
    0.25,0.25,0.25) (If num_vals_x !=  num_vals_y, this assumes that
    num_vals_x is the smaller of the two.)

    References
    ----------
    1. R.R. Tucci, "A New  Algorithm for Calculating Squashed Entanglement
    and a Python Implementation Thereof"


    Attributes
    ----------
    num_rows : int
        equals product(row_shape)
    num_vals_min : int
        equals min( num_vals_x, num_vals_y)
    num_vals_x : int
        equals product(row_shape_x)
    num_vals_y : int
        equals product(row_shape_y)
    row_shape : tuple[int]
    row_shape_x : tuple[int]
        subset of row_shape, only items indexed by x_axes
    row_shape_y : tuple[int]
        subset of row_shape, only items indexed by y_axes
    x_axes : list{int]
    y_axes : list{int]

    """

    def __init__(self, num_rows, row_shape, x_axes, y_axes):
        """
        Constructor

        Parameters
        ----------
        num_rows : int
        row_shape : tuple[int]
        x_axes : list{int]
        y_axes : list{int]
        """

        self.num_rows = num_rows
        self.row_shape = row_shape
        self.x_axes = x_axes
        self.y_axes = y_axes

        assert num_rows == np.product(np.array(row_shape))
        assert len(row_shape) == len(set(x_axes).union(set(y_axes)))

        self.row_shape_x = tuple([row_shape[k] for k in x_axes])
        self.row_shape_y = tuple([row_shape[k] for k in y_axes])
        self.num_vals_x = np.product(np.array(self.row_shape_x))
        self.num_vals_y = np.product(np.array(self.row_shape_y))
        self.num_vals_min = min(self.num_vals_x, self.num_vals_y)

    def get_st_vec(self):
        """
        This method returns the state vector of the state.

        Returns
        -------
        np.ndarray
            shape=(self.num_rows,)

        """
        st_vec = np.zeros(shape=(self.num_vals_x, self.num_vals_y),
                          dtype=complex)

        for ix in range(self.num_vals_min):
            st_vec[ix, ix] = 1/np.sqrt(self.num_vals_min)
        st_vec = st_vec.reshape(self.row_shape_x + self.row_shape_y)
        st_vec = np.transpose(st_vec, self.x_axes + self.y_axes)
        st_vec = st_vec.reshape((self.num_rows,))
        return st_vec

    def get_known_entang(self):
        """
        This method returns the known entanglement of the state, i.e. log(
        self.num_vals_min)

        Returns
        -------
        float

        """
        return np.log(self.num_vals_min)


if __name__ == "__main__":
    from entanglish.EntangCase import *
    from entanglish.PureStEnt import *

    def main():
        dm_max = DenMat(24, (2, 2, 3, 2))
        max_ent_st = MaxEntangState(dm_max.num_rows, dm_max.row_shape,
                                    [0, 1, 3], [2])
        EntangCase.check_max_entang_st(max_ent_st)
        st_vec = max_ent_st.get_st_vec()
        entang = max_ent_st.get_known_entang()
        dm_max.set_arr_from_st_vec(st_vec)

        print('st_vec=\n', st_vec)
        print("entang=", entang)
        ecase = PureStEnt(dm_max, 'eigen')
        pf = ecase.get_entang_profile()
        ecase.print_entang_profiles([pf], dm_max.row_shape)
    main()

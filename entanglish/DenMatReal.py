from entanglish.DenMat import *


class DenMatReal:
    """
    BT = bottom triangle without diag, TT = top triangle with diag

    Density matrices are Hermitian matrices. This class stores a Hermitian
    matrix in an array of dtype=float contrary to DenMat which uses an array
    of dtype=complex. In this class, the real part (resp., imag part) of the
    Hermitian matrix is stored in TT (resp., BT) of real array

    Attributes
    ----------
    num_rows : int
    re_arr : np.ndarray
        real array of shape (num_rows, num_rows)
    row_shape : int

    """
    def __init__(self, num_rows, row_shape, re_arr=None):
        """
        Constructor

        Parameters
        ----------
        num_rows : int
        row_shape : tuple[int]
        re_arr : np.ndarray

        Returns
        -------


        """
        self.num_rows = num_rows
        assert self.num_rows > 0

        self.row_shape = row_shape
        assert num_rows == ut.scalar_prod(row_shape)

        self.re_arr = re_arr
        if re_arr is not None:
            assert re_arr.shape == (num_rows, num_rows)

    @staticmethod
    def r2c(re_arr):
        """
        real to complex. This method maps a real array to a complex array.

        Parameters
        ----------
        re_arr : np.ndarray

        Returns
        -------
        np.ndarray

        """
        assert re_arr.dtype == float
        assert re_arr.ndim == 2
        assert re_arr.shape[0] == re_arr.shape[1]
        arr = np.zeros(re_arr.shape, dtype=complex)
        for row in range(arr.shape[0]):
            for col in range(arr.shape[0]):
                if col == row:
                    arr[row, col] = re_arr[row, col]
                elif col > row:
                    arr[row, col] = re_arr[row, col] + 1j*re_arr[col, row]
                else:
                    arr[row, col] = re_arr[col, row] - 1j*re_arr[row, col]
        return arr

    @staticmethod
    def c2r(arr):
        """
        complex to real. This method maps a complex array to a real array.

        Parameters
        ----------
        arr : np.ndarray

        Returns
        -------
        np.ndarray

        """
        assert arr.dtype == complex
        assert arr.ndim == 2
        assert arr.shape[0] == arr.shape[1]
        re_arr = np.zeros(arr.shape, dtype=float)
        for row in range(arr.shape[0]):
            for col in range(arr.shape[0]):
                if col == row:
                    re_arr[row, col] = arr[row, col].real
                elif col > row:
                    re_arr[row, col] = \
                        (arr[row, col] + arr[col, row]).real/2
                else:
                    re_arr[row, col] = \
                        (-arr[row, col] + arr[col, row]).imag/2
        return re_arr

    def __getitem__(self, key):
        """
        If obj is an object of this class, this makes
        obj[j, k] = obj.re_arr[j, k]

        Parameters
        ----------
        key : tuple[int, int]

        Returns
        -------
        int|float

        """
        return self.re_arr[key]

    def __setitem__(self, key, item):
        """
        If obj is an object of this class, this makes
        obj[j, k] = 5 the same as obj.re_arr[j, k] = 5

        Parameters
        ----------
        key : tuple
        item : int|float

        Returns
        -------
        None

        """
        self.re_arr[key] = item

    def __str__(self):
        """
        This method returns str(self.re_arr)

        Returns
        -------
        str

        """
        return str(self.re_arr)

if __name__ == "__main__":
    def main():
        dm = DenMat(3, (3,))
        evas = np.array([.1, .2, .7])
        dm.set_arr_to_rand_den_mat(evas)

        re_dm = DenMatReal(dm.num_rows, dm.row_shape)
        re_dm.re_arr = DenMatReal.c2r(dm.arr)

        print('dm\n', dm)
        print('re_dm\n', re_dm)
        dm_arr = DenMatReal.r2c(DenMatReal.c2r(dm.arr))
        print('dist(dm_arr, dm.arr)=', np.linalg.norm(dm_arr-dm.arr))
    main()

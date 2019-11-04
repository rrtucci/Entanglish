import numpy as np
import entanglish.utilities as ut
import copy as cp
import scipy as sc
import scipy.linalg as la


class DenMat:
    """
    This class specifies a quantum mechanical Density Matrix. The class also
    performs a large number of operations on the density matrix self.

    Normally, a density matrix is a Hermitian matrix whose eigenvalues
    define a probability distribution (i.e., the eigenvalues are real
    numbers between zero and one and they sum to 1.) This class does not
    automatically check that self is Hermitian, although you can do it
    yourself with the method is_hermitian(). The class doesn't check that
    the trace is one either, although you can do it yourself with the method
    trace(). It is often useful to use this class to hold an "un-normalized"
    density matrix whose trace is not 1. So trace=1 is seldom assumed by the
    functions of this class. On the other hand, many of the functions in
    this class, especially the ones that involve finding eigenvalues,
    do assume that self is Hermitian.

    The attribute arr holds a numpy array of shape (num_rows, num_rows),
    where num_rows is also an attribute. Another attribute is row_shape,
    which is a tuple of integers whose product is num_rows. At any time,
    arr can be reshaped from (num_rows, num_rows) to row_shape*2.

    For example, suppose row_shape = (3, 4, 2). Then this density matrix
    will be taken to represent a mixed state of 3 qudits with d=3, d=4 and
    d=2 in that order. When we take a partial trace over the second (d=4)
    qudit, we contract the second row index with the second column index.

    To use this class in conjunction with Qubiter, replace self.arr in this
    class with the output of Qubiter.StateVec.get_den_mat()

    The marginals of self is defined as a list of partial traces of self.
    The n'th item in the list of marginals is the partial trace of self,
    traced over all qudits except the n'th. The Kronecker product of the
    marginals of self is a "separable" density matrix, in the sense that
    there is no correlation among its qudits.

    Attributes
    ----------
    arr : np.ndarray
        numpy array of shape (num_rows, num_rows) which contains entries of
        a density matrix
    marginals : list[DenMat]
    num_rows : int
    row_shape : tuple[int]
        tuple of integers whose product is num_rows

    """
    def __init__(self, num_rows, row_shape, arr=None):
        """
        Constructor


        Parameters
        ----------
        num_rows : int
        row_shape : tuple[int]
        arr : np.ndarray
            shape=(num_rows, num_rows)

        Returns
        -------


        """

        self.num_rows = num_rows
        assert self.num_rows > 0

        self.row_shape = row_shape
        # print('---...', row_shape)
        assert num_rows == ut.scalar_prod(row_shape)

        self.arr = arr
        if arr is not None:
            assert arr.shape == (num_rows, num_rows)

        # calculate marginals only if they they are needed
        self.marginals = None

    def set_marginals(self):
        """
        This method calculates the single-axis marginals of self and stores
        them in self.marginals.

        Returns
        -------
        None

        """
        num_row_axes = len(self.row_shape)
        if num_row_axes == 1:
            marginals = [self.copy()]
        else:
            marginals = []
            for k in range(num_row_axes):
                traced_axes_set = set(
                    [ax for ax in range(num_row_axes) if ax != k])
                marginals.append(self.get_partial_tr(traced_axes_set))
        self.marginals = marginals

    def copy(self):
        """
        This method returns a copy of self.

        Returns
        -------
        DenMat

        """
        return DenMat(self.num_rows, self.row_shape, cp.copy(self.arr))

    @staticmethod
    def new_const_den_mat(num_rows, row_shape, const):
        """
        This method returns a DenMat object with an arr which is a diagonal
        matrix with 'const' in its diagonal.

        Parameters
        ----------
        num_rows : int
        row_shape : tuple[int]
        const : int|float|complex

        Returns
        -------
        DenMat

        """
        arr = np.diag(np.array([const]*num_rows))
        return DenMat(num_rows, row_shape, arr)

    def set_arr_to_zero(self):
        """
        Sets self.arr to zero matrix.

        Returns
        -------
        None

        """
        self.arr = np.zeros(
            shape=(self.num_rows, self.num_rows), dtype=complex)

    def set_arr_to_rand_den_mat(self, evas):
        """
        evas = eigenvalues. 
        
        This method sets self.arr to a random density matrix UDU^dag,
        where U is a random unitary matrix, D is a non-random diagonal
        matrix with diagonal equal to the input 1D array 'evas', and U^dag
        is the Hermitian conjugate of U. This method checks that evas is a
        probability distribution. The DenMat returned by this method is
        useful for testing purposes.

        Parameters
        ----------
        evas : np.ndarray
            evas stands for eigenvalues. 1D array of floats, of shape (
            self.num_rows, )

        Returns
        -------
        None

        """
        assert ut.is_prob_dist(evas)
        umat = ut.random_unitary(len(evas))
        umat_H = umat.conj().T
        # multiply each col of umat by corresponding eigenvalue
        for k in range(len(evas)):
            umat[:, k] *= evas[k]
        self.arr = np.dot(umat, umat_H)

    def set_arr_from_st_vec(self, st_vec):
        """
        Sets self.arr to st_vec*set_vec^dag where st_vec is a column vector 
        of shape (num_rows, ). For qubits, st_vec should be a traditional 
        state vector, meaning that when reshaped to [2]*num_bits, 
        the components are .. |s_2>|s_1>|s_0> where s_i = 0, 1 corresponds 
        to the i'th qubit. s_0 is last so ZL (Zero Last) convention. 

        Parameters
        ----------
        st_vec : np.ndarray
            shape=(self.num_rows,)

        Returns
        -------
        None

        """
        assert st_vec.shape == (self.num_rows,)
        self.arr = np.outer(st_vec, np.conj(st_vec))

    def add_const_to_diag_of_arr(self, const):
        """
        Adds constant `const` to diagonal of self.arr.

        Parameters
        ----------
        const : complex|float|int

        Returns
        -------
        None

        """
        self.arr[np.diag_indices_from(self.arr)] += const

    def add_vec_to_diag_of_arr(self, vec_arr):
        """
        Adds vector `vec_arr` to diagonal of self.arr

        Parameters
        ----------
        vec_arr : np.ndarray

        Returns
        -------
        None

        """
        assert vec_arr.shape == (self.num_rows,), \
            str(vec_arr.shape) + ' is not ' + str(self.num_rows)
        self.arr[np.diag_indices_from(self.arr)] += vec_arr

    def replace_diag_of_arr(self, new_diag):
        """
        Replaces diagonal of self.arr by new_diag.

        Parameters
        ----------
        new_diag : np.ndarray

        Returns
        -------
        None

        """
        assert new_diag.shape == (self.num_rows, ),\
            str(new_diag.shape) + ' is not ' + str(self.num_rows)

        self.arr[np.diag_indices_from(self.arr)] = new_diag

    def normalize_diag_of_arr(self):
        """
        Divides the diagonal of self.arr by the trace of self.arr.

        Returns
        -------
        None

        """
        tr = np.trace(self.arr).real
        assert abs(tr) > 1e-6
        self.arr[np.diag_indices_from(self.arr)] /= tr

    def normalize_entire_arr(self):
        """
        Divides all of self.arr by the trace of self.arr.

        Returns
        -------
        None

        """
        tr = np.trace(self.arr).real
        assert abs(tr) > 1e-6
        self.arr /= tr

    @staticmethod
    def get_kron_prod_of_den_mats(den_mat_list):
        """
        Takes as input a list of DenMat's and returns a DenMat which is the
        Kronecker product of the items in the input list.

        Parameters
        ----------
        den_mat_list : list[DenMat]

        Returns
        -------
        DenMat

        """
        row_shape = []
        for dm in den_mat_list:
            row_shape += list(dm.row_shape)
        row_shape = tuple(row_shape)
        num_rows = ut.scalar_prod(row_shape)
        arr = ut.kron_prod([dm.arr for dm in den_mat_list])
        return DenMat(num_rows, row_shape, arr)

    @staticmethod
    def new_with_permuted_qudits(dm, perm):
        """
        This method returns a DenMat in which the rows (and columns) of
        dm.arr have been permuted according to perm.

        Parameters
        ----------
        dm : DenMat
        perm : list[int}
            perm is a permutation of range(len(self.row_shape))

        Returns
        -------
        DenMat

        """
        nrows = dm.num_rows
        sorted_perm = sorted(perm)
        sorted_perm1 = list(range(len(dm.row_shape)))
        assert sorted_perm1 == sorted_perm,\
            'got:' + str(sorted_perm) + ', but expected:' +\
                str(sorted_perm1)
        displaced_perm = [k + len(perm) for k in perm]
        new_row_shape = tuple([dm.row_shape[k] for k in perm])
        new_arr = cp.copy(dm.arr)
        new_arr = new_arr.reshape(tuple(list(dm.row_shape)*2))
        # print(",,,", new_arr.shape)
        new_arr = np.transpose(new_arr, perm + displaced_perm)
        new_arr = new_arr.reshape((nrows, nrows))
        return DenMat(nrows, new_row_shape, new_arr)

    def get_rho_xy(self, x_axes, y_axes):
        """
        The inputs 'x_axes' and 'y_axes' must be mutually exclusive lists 
        whose union, after sorting, is range(len(self.row_shape)), 
        which equals [0, 1, 2, ..., number of row axes -1]. The output is a
        DenMat in which the rows (and columns) of self.arr have been 
        permuted to the order x_axes + y_axes. 

        Parameters
        ----------
        x_axes : list[int]
        y_axes : list[int]

        Returns
        -------
        DenMat

        """
        return DenMat.new_with_permuted_qudits(self, x_axes + y_axes)

    def get_partial_tr(self, traced_axes_set):
        """
        This method returns the partial trace of a density matrix den_mat.
        It traces over the indices (a.k.a. axes) in the non-empty set
        traced_axes_set. To get a full trace, just do den_mat.trace()

        Parameters
        ----------
        traced_axes_set : set[int]
             Set of axes being traced over

        Returns
        -------
        DenMat

        """
        # This method is similar to Qubiter.StatVec.get_partial_tr()
        num_row_axes = len(self.row_shape)
        traced_axes = list(traced_axes_set)
        assert all(0 <= x < num_row_axes for x in traced_axes)
        assert 0 < len(traced_axes) < num_row_axes, \
            "Tracing over zero or all qudits. " + \
            "To trace over all qudits, just do den_mat.trace()."
        untraced_axes = [k for k in range(num_row_axes)
                         if k not in traced_axes]

        new_row_shape = tuple([self.row_shape[ax] for ax in untraced_axes])
        new_num_rows = ut.scalar_prod(new_row_shape)

        arr = cp.copy(self.arr)
        arr = arr.reshape(tuple(list(self.row_shape)*2))
        num_traces = len(traced_axes)
        # print('/..', arr.shape)
        for k in range(num_traces):
            # print(',,,..', traced_axes)
            ax = traced_axes.pop(0)
            arr = np.trace(arr, axis1=ax, axis2=ax + num_row_axes - k)
            # print('//', arr.shape)
            traced_axes = list(map(lambda x: (x if x <= ax else x-1),
                                   traced_axes))
        # print('///', arr.shape)
        arr = arr.reshape((new_num_rows, new_num_rows))
        return DenMat(new_num_rows, new_row_shape, arr)

    def get_set_of_all_other_axes(self, axes_set):
        """
        This method returns the complement set wrt range(len(self.row_shape))
        of the set of axes axes_set

        Parameters
        ----------
        axes_set : set[int]

        Returns
        -------
        set[int]

        """

        all_axes = range(len(self.row_shape))
        comp_axes_set = set([ax for ax in all_axes if ax not in axes_set])
        return comp_axes_set

    def get_entropy(self, method='eigen'):
        """
        This method returns an exact entropy of density matrix self. Uses
        natural log for entropy. Assumes eigenvalues of self are
        non-negative and sum to 1.

        Parameters
        ----------
        method : str        
            method used to calculate log of array. Either 'eigen' or 'pade' 

        Returns
        -------
        float

        """
        ent = 0.0
        if method == 'eigen':
            evas = np.real(np.linalg.eigvalsh(self.arr))
            ent = ut.get_entropy_from_probs(evas)
        elif method == 'pade':
            ent = - np.trace(np.dot(self.arr, la.logm(self.arr))).real
        else:
            assert False, 'unsupported method for ' +\
                          'calculating entropy of a density matrix.'

        return ent

    def get_mutual_info(self, traced_axes_set, method='eigen'):
        """
        This method returns the mutual information for x_axes = list(
        traced_axes_set) and y_axes = row axes not in x_axes. Uses natural
        log for entropy. Assumes eigenvalues of self are non-negative and
        sum to 1.

        Parameters
        ----------
        traced_axes_set : set[int]
        method : str        
            method used to calculate log of array. Either 'eigen' or 'pade' 

        Returns
        -------
        float

        """
        num_row_axes = len(self.row_shape)
        x_axes = list(traced_axes_set)
        assert all(0 <= x < num_row_axes for x in x_axes)
        y_axes = [k for k in range(num_row_axes) if k not in x_axes]

        dm_x = self.get_partial_tr(set(y_axes))
        dm_y = self.get_partial_tr(set(x_axes))
        mi = - self.get_entropy(method) \
             + dm_x.get_entropy(method) \
             + dm_y.get_entropy(method)
        return mi

    def dm_op(self, right, arr_op):
        """
        Internal method used in magic methods __add__, __sub__, __mul__, 
        which define binary operations between self and another DenMat. 

        Parameters
        ----------
        right : DenMat
        arr_op : wrapper_descriptor
            This is going to be either
            np.ndarray.[__add__, __sub__], np.dot

        Returns
        -------
        DenMat

        """
        assert self.num_rows == right.num_rows
        assert self.row_shape == right.row_shape
        new_arr = arr_op(self.arr, right.arr)
        return DenMat(self.num_rows, self.row_shape, new_arr)

    def dm_iop(self, right, arr_iop):
        """
        Internal method used in magic methods __iadd__, __isub__, __imul__, 
        which define in-place binary operations between self and another 
        DenMat. 

        Parameters
        ----------
        right : DenMat
        arr_iop : wrapper_descriptor
            This is going to be either
            np.ndarray.[__iadd__, __isub__], np.dot

        Returns
        -------
        DenMat
            returns self

        """
        assert self.num_rows == right.num_rows
        assert self.row_shape == right.row_shape
        arr_iop(self.arr, right.arr)
        return self

    def __add__(self, right):
        """
        Defines '+' between self and another DenMat, the input 'right'.

        Parameters
        ----------
        right : DenMat

        Returns
        -------
        DenMat

        """
        return self.dm_op(right, np.ndarray.__add__)

    def __iadd__(self, right):
        """
        Defines '+=' between self and another DenMat, the input 'right'.

        Parameters
        ----------
        right : DenMat

        Returns
        -------
        DenMat

        """
        return self.dm_iop(right, np.ndarray.__iadd__)

    def __sub__(self, right):
        """
        Defines '-' between self and another DenMat, the input 'right'.

        Parameters
        ----------
        right : DenMat

        Returns
        -------
        DenMat

        """
        return self.dm_op(right, np.ndarray.__sub__)

    def __isub__(self, right):
        """
        Defines '-=' between self and another DenMat, the input 'right'.

        Parameters
        ----------
        right : DenMat

        Returns
        -------
        DenMat

        """
        return self.dm_iop(right, np.ndarray.__isub__)

    def __mul__(self, right):
        """
        Defines '*' as np.dot() between self and another DenMat or a scalar
        (either int, float or complex), the input 'right'. Multiplication by
        a scalar is defined only if scalar is on the right side of self.

        Parameters
        ----------
        right : DenMat|complex|float|int

        Returns
        -------
        DenMat

        """
        if isinstance(right, (int, float, complex)):
            new_arr = self.arr * right
            return DenMat(self.num_rows, self.row_shape, new_arr)
        else:
            return self.dm_op(right, np.dot)

    def __imul__(self, right):
        """
        Defines '*=' as inplace matrix multiplication between self and
        another DenMat or a scalar (either int, float or complex), the input
        'right'.

        Parameters
        ----------
        right : DenMat|complex|float|int

        Returns
        -------
        DenMat

        """
        if isinstance(right, (int, float, complex)):
            self.arr *= right
            return self
        else:
            return self.dm_op(right, np.dot)

    def __getitem__(self, key):
        """
        Defines self[int1, int2] to be same as self.arr[int1, int2]

        Parameters
        ----------
        key : tuple[int, int]

        Returns
        -------
        complex

        """
        return self.arr[key]

    def __setitem__(self, key, item):
        """
        Defines assignment `self[int1, int2] = item` to be same as
        `self.arr[int1, int2] = item`.

        Parameters
        ----------
        key : tuple[int, int]
        item : complex|float|int

        Returns
        -------
        None

        """
        self.arr[key] = item

    def __str__(self):
        """
        Returns str(self.arr).

        Returns
        -------
        str

        """
        return str(self.arr)

    def __repr__(self):
        """
        Returns str(self.arr).

        Returns
        -------
        str

        """
        return self.__str__()

    def is_pure_state(self):
        """
        This method returns a bool which answers the question whether self
        is a pure state or not.

        Returns
        -------
        bool

        """
        return np.linalg.norm(np.dot(self.arr, self.arr) - self.arr) < 1e-6

    def depurify(self, eps):
        """
        If self is a pure state, this method returns a nearby density mat
        that is mixed

        Parameters
        ----------
        eps : float
            small positive number

        Returns
        -------
        DenMat

        """
        probs = np.random.random(self.num_rows)
        probs = probs/np.sum(probs)
        arr = (self.arr + eps*np.diag(probs)) / (self.trace() + eps)
        return DenMat(self.num_rows, self.row_shape, arr)

    def herm(self):
        """
        This method returns a DenMat which is the Hermitian conjugate of self.

        Returns
        -------
        DenMat

        """
        return DenMat(self.num_rows, self.row_shape, self.arr.conj().T)

    def conj(self):
        """
        This method returns a DenMat which is the complex conjugate of self.

        Returns
        -------
        DenMat

        """
        return DenMat(self.num_rows, self.row_shape, self.arr.conj())

    def trace(self):
        """
        This method returns the real part of the full trace of self.

        Returns
        -------
        float

        """
        im = np.trace(self.arr).imag
        assert im < 1e-4, 'imag=' + str(im)
        return np.trace(self.arr).real

    def norm(self):
        """
        This method returns the 2-norm of self.arr

        Returns
        -------
        float

        """
        return np.linalg.norm(self.arr)

    def sqrt(self, method='eigen'):
        """
        This method returns a DenMat which is the matrix square root of self.

        Parameters
        ----------
        method : str
            method used to calculate sqrt. Either 'eigen' or 'pade'.

        Returns
        -------
        DenMat

        """
        sqrtm_arr = None
        if method == 'eigen':
            sqrtm_arr = ut.fun_of_herm_arr(lambda x: np.sqrt(x) if x > 0 else
            0, self.arr)
        elif method == 'pade':
            sqrtm_arr = la.sqrtm(self.arr)
        else:
            assert False, 'unsupported method'
        return DenMat(self.num_rows, self.row_shape, sqrtm_arr)

    def exp(self, method='eigen'):
        """
        This method returns a DenMat which is the matrix exponential of self.

        Parameters
        ----------
        method : str
            method used to calculate exp. Either 'eigen' or 'pade'.

        Returns
        -------
        DenMat

        """
        expm_arr = None
        if method == 'eigen':
            expm_arr = ut.fun_of_herm_arr(np.exp, self.arr)
        elif method == 'pade':
            expm_arr = la.expm(self.arr)
        else:
            assert False, 'unsupported method'
        return DenMat(self.num_rows, self.row_shape, expm_arr)

    def log(self, method='eigen', clipped=True, eps=1e-4,
            clip_to_zero=False):
        """
        This method returns a DenMat which is the matrix natural log of self.

        Parameters
        ----------
        method : str
            method used to calculate the natural log. Either 'eigen' or 'pade'.
        clipped : bool
            clips logs (see ut.clipped_log_of_vec) iff this is True
        eps : float
            used only if clipping log
        clip_to_zero : bool
            used only if clipping log

        Returns
        -------
        DenMat

        """
        # self.add_const_to_diag_of_arr(1e-8)
        # self.normalize_diag_of_arr()
        logm_arr = None
        if method == 'eigen':
            if clipped:
                logm_arr = ut.fun_of_herm_arr(ut.clipped_log_of_vec,
                                              self.arr,
                                              eps=eps,
                                              clip_to_zero=clip_to_zero)
            else:
                logm_arr = ut.fun_of_herm_arr(np.log,
                                              self.arr)

        elif method == 'pade':
            logm_arr = la.logm(self.arr)
        else:
            assert False, 'unsupported method'
        return DenMat(self.num_rows, self.row_shape, logm_arr)

    def positive_part(self, threshold=1e-5):
        """
        This method returns a DenMat in which negative (< 0) eigenvalues of
        self.arr are replaced by zero.

        Returns
        -------
        DenMat

        """
        fun = ut.positive_part_of_vec
        pos_arr = ut.fun_of_herm_arr(fun, self.arr)
        return DenMat(self.num_rows, self.row_shape, pos_arr)

    def purer_version(self):
        """
        This method returns a DenMat in which all eigenvalues of self.arr
        except the maximum one are replaced by zero.

        Returns
        -------
        DenMat

        """
        evas, evec_cols = np.linalg.eigh(self.arr)
        max_pos = np.argmax(evas)
        vec = evec_cols[:, max_pos]
        arr = np.outer(vec, np.conj(vec))*evas[max_pos]
        return DenMat(self.num_rows, self.row_shape, arr)

    def purer_version2(self):
        """
        This method returns a DenMat in which self.arr is replaced by (
        self.arr^2)/sqrt(tr2) where tr2 is the trace of self.arr^2. This
        transformation has no effect on self if self is an un-normalized
        pure state.

        Returns
        -------
        DenMat

        """
        dm2 = self*self
        tr2 = dm2.trace()
        return dm2*(1/np.sqrt(tr2))

    def inv(self, regulator=0.0):
        """
        This method returns a DenMat which is the inverse matrix of self.

        logs, exponentials and sqrt's of matrices are calculated by various
        methods. Inverses of matrices, on the other hand, are calculated a
        single way.

        Parameters
        ----------
        regulator : float
            this constant is added to diagonal of copy of self.arr before
            taking the inverse of copy

        Returns
        -------
        DenMat

        """
        arr = cp.copy(self.arr)
        arr[np.diag_indices_from(arr)] += regulator
        return DenMat(self.num_rows, self.row_shape, np.linalg.inv(arr))

    def pseudo_inv(self, eps=1e-5):
        """
        This method returns a DenMat which is the Penrose pseudo inverse
        matrix of self. By pseudo inverse, we mean that it takes the inverse
        of non-zero (abs > eps) eigenvalues only, but sets those eigenvalues
        with abs < eps to exactly zero.

        Parameters
        ----------
        eps : float

        Returns
        -------
        DenMat

        """
        pseudo_inv_arr = ut.fun_of_herm_arr(
            lambda x: 1/x if abs(x) > eps else 0, self.arr)
        return DenMat(self.num_rows, self.row_shape, pseudo_inv_arr)

    def get_eigenvalue_proj_ops(self, eps=1e-5):
        """
        This method returns a tuple of 2 DenMat that carry Hermitian
        projection operators:

        proj_0 projects out the space of zero (abs < eps) eigenvalues. It is
        obtained by replacing in the eigen decomposition of self.arr,
        zero eigenvalues (abs < eps) by 1 and non-zero ones (abs > eps) by 0.

        proj_1 projects out the space of non-zero (abs > eps) eigenvalues.
        It is obtained by replacing in the eigen decomposition of self.arr,
        zero eigenvalues (abs < eps) by 0 and non-zero ones (abs > eps) by 1.

        Parameters
        ----------
        eps : float
            0 < eps << 1

        Returns
        -------
        DenMat, DenMat

        """

        assert ut.is_hermitian_arr(self.arr)
        evas, evec_cols = np.linalg.eigh(self.arr)
        is_zero = []
        is_not_zero = []
        for x in evas:
            if abs(x) < eps:
                is_zero.append(1)
                is_not_zero.append(0)
            else:
                is_zero.append(0)
                is_not_zero.append(1)
        arr_proj_zero = ut.fun_of_herm_arr_from_eigen_sys(
            lambda xx: xx, np.array(is_zero), evec_cols)

        arr_proj_not_zero = ut.fun_of_herm_arr_from_eigen_sys(
            lambda xx: xx, np.array(is_not_zero), evec_cols)

        proj_0 = DenMat(self.num_rows, self.row_shape, arr_proj_zero)
        proj_1 = DenMat(self.num_rows, self.row_shape, arr_proj_not_zero)
        return proj_0, proj_1

    def get_eigen_sys_of_marginals(self):
        """
        eva = eigenvalue, evec = eigenvector, sep = separable.

        This method assumes self is Hermitian.

        We will call an 'eigensystem' of a density matrix: a tuple, whose
        first item is a 1D numpy array, call it evas, that carries the
        eigenvalues of the density matrix, and the second item is a 2D numpy
        array, call it eigen_cols, whose i'th columns is an eigenvector for
        the i'th eigenvalue (i.e., evas[i]) of the density matrix.

        This method returns a tuple of two lists. The first list contains
        the first part (eigenvalues) of the eigensystem of each marginal of
        self. The second list contains the second part (eigenvectors) of the
        eigensystem of each marginal of self.

        Returns
        -------
        tuple[list[np.ndarray], list[np.ndarray]]

        """
        vec_list = []
        mat_list = []
        assert self.marginals
        for marg in self.marginals:
            # vec_evas = vector with eigenvalues
            # mat_evecs = unitary matrix with eigenvectors as columns
            vec_evas, mat_evecs = np.linalg.eigh(marg.arr)
            vec_list.append(vec_evas)
            # print("------------", vec_evas)
            mat_list.append(mat_evecs)
        return vec_list, mat_list

    def switch_arr_basis(self, umat, reverse=False):
        """
        This method returns a new DenMat whose arr is U^dag(self.arr)U (or
        the reverse, U(self.arr)U^dag, if the input bool parameter 'reverse'
        is set to True.) U = umat , U^dag = Hermitian conjugate of U

        Parameters
        ----------
        umat : np.ndarray
        reverse : bool

        Returns
        -------
        DenMat

        """

        new_arr = ut.switch_arr_basis(self.arr, umat, reverse)
        return DenMat(self.num_rows, self.row_shape, new_arr)

    @staticmethod
    def get_fun_of_dm_from_eigen_sys(num_rows, row_shape,
                                     eigen_sys, fun_of_scalars,
                                     **fun_kwargs):
        """
        If (evas, U) = eigen_sys and fun = fun_of_scalars, then this method
        returns U.fun(evas).U^dag, where U^dag is the Hermitian conjugate of
        the unitary matrix U.

        The function calculated (for example, np.exp, np.log, etc.) is
        passed in as the input 'fun_of_scalars'. To get just an approx to dm
        instead of an approx to fun of dm, use fun_of_scalars = lambda x: x

        Parameters
        ----------
        num_rows : int
        row_shape : tuple[int]
        eigen_sys : tuple[np.ndarray, np.ndarray]
        fun_of_scalars :
            function that can act on scalars or numpy arrays element-wise
        fun_kwargs : dict
            dict of keyword args that fun depends on

        Returns
        -------
        DenMat

        """
        evas, evec_cols = eigen_sys
        arr = ut.fun_of_herm_arr_from_eigen_sys(
            fun_of_scalars, evas, evec_cols, **fun_kwargs)
        return DenMat(num_rows, row_shape, arr)


if __name__ == "__main__":
    def main():
        dm1 = DenMat(3, (3,))

        dm = DenMat.new_const_den_mat(dm1.num_rows, dm1.row_shape, 2)
        print('constant 2\n', dm)

        dm1.set_arr_to_zero()
        print('set_arr_to_zero\n', dm1)

        evas = np.array([.1, .3, .6])
        dm1.set_arr_to_rand_den_mat(evas)
        print('set_arr_to_rand_den_mat\n', dm1)
        print('out_evas - in_evas\n', np.linalg.eigvalsh(dm1.arr) - evas)

        dm1.set_arr_from_st_vec(np.array([1+.1j, 2, 3]))
        print('set_arr_from_st_vec\n', dm1)

        dm1.set_arr_to_zero()
        dm1.add_const_to_diag_of_arr(1)
        print('add_const_to_diag_of_arr\n', dm1)

        dm1.add_vec_to_diag_of_arr(np.array([1, 2, 3]))
        print('add_vec_to_diag_of_arr\n', dm1)

        dm1.replace_diag_of_arr(np.array([.1, .2, .3]))
        print('replace_diag_of_arr\n', dm1)

        dm1.normalize_diag_of_arr()
        print('normalize_diag_of_arr\n', dm1)

        dm1 *= 3
        dm1.normalize_entire_arr()
        print('normalize entire\n', dm1)

        dm1 = DenMat(2, (2,))
        dm1.arr = np.array([[.5, .2], [.2, .5]])
        print('1 qubit dm\n', dm1)

        id2 = DenMat(2, (2,))
        id2.arr = np.array([[1, 0], [0, 1]])
        print('id2\n', id2)

        dm2 = DenMat.get_kron_prod_of_den_mats([dm1, id2*.5, dm1, id2*.5])
        # print('4 fold kron of rot_y\n', dm2)
        dm2.set_marginals()
        print("init marginals dm1,id2, dm1, id2\n", dm2.marginals)

        rho_xy = dm2.get_rho_xy([0, 2], [1, 3])
        # print('rho_02,13\n', rho_xy)
        rho_xy.set_marginals()
        print("permuted marginals dm1, dm1, id2, id2\n", rho_xy.marginals)

        rho_x = rho_xy.get_partial_tr({1, 3})
        print('rho_02\n', rho_x)

        print('rho_0 - dm1\n', rho_x.get_partial_tr({1}) - dm1)
        print('rho_2 - id2*.5\n', rho_x.get_partial_tr({0}) - id2*.5)

        print('entropy of rho_x=', rho_x.get_entropy())

        print('mutual info of rho_xy=', rho_xy.get_mutual_info({0, 2}))

        print('rho_xy - dm2\n', rho_xy-dm2)
        print('rho_xy + dm2\n', rho_xy+dm2)
        print('rho_x*inv_rho_x\n', rho_x*rho_x.inv())
        print('rho_xy[0, 0]=', rho_xy[0, 0])
        print('rho_x-log exp rho_x\n', rho_x - rho_x.log().exp())
        print('rho_x*pinv_rho_x\n', rho_x*rho_x.pseudo_inv())

    main()






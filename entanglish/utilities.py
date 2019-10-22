from functools import reduce
import numpy as np
import copy as cp
from math import factorial


def scalar_prod(scalars_list):
    """
    This method returns the product of the list of scalars which it has as
    input.

    Parameters
    ----------
    scalars_list : list[int|float|complex] | tuple[int|float|complex]

    Returns
    -------
    complex|float|int

    """
    if len(scalars_list) == 1:
        return scalars_list[0]
    else:
        return reduce(lambda x, y: x*y, scalars_list)


def kron_prod(mat_list):
    """
    This method returns the Kronecker product of the list of matrices which
    is has as input.

    Parameters
    ----------
    mat_list : list[np.ndarray]

    Returns
    -------
    np.ndarray

    """
    num_mats = len(mat_list)
    prod = mat_list[0]
    for k in range(1, num_mats, 1):
        prod = np.kron(prod, mat_list[k])
    return prod


def mat_elem(v1, a, v2):
    """
    This method returns the matrix element <v1|a|v2>, where v1 and v2 are
    column vectors and 'a' a matrix.

    Parameters
    ----------
    v1 : np.ndarray
    a : np.ndarray
    v2 : np.ndarray

    Returns
    -------
    complex


    """
    return np.dot(np.dot(v1.conj().T, a), v2)


def switch_arr_basis(arr, umat, reverse=False):
    """
    This method takes as input a square array 'arr' and returns a new array
    which is a similarity transformation U^dag(arr)U of 'arr' that changes
    the basis of arr from inbasis to sbasis (or the reverse, U(arr)U^dag,
    from sbasis to inbasis if the input bool parameter 'reverse' is set to
    True.) U = umat , U^dag = Hermitian conjugate of U

    Parameters
    ----------
    arr : np.ndarray
    umat : np.ndarray
    reverse : bool

    Returns
    -------
    np.ndarray

    """
    umat_H = umat.conj().T
    assert arr.shape == umat.shape
    if not reverse:
        new_arr = np.dot(np.dot(umat_H, arr), umat)
    else:
        new_arr = np.dot(np.dot(umat, arr), umat_H)
    return new_arr


def clip(x, limits):
    """
    This method clips x between limits[0] and limits[1]

    Parameters
    ----------
    x : int|float
    limits : list[int|float]

    Returns
    -------
    int|float

    """
    return min(max(limits[0], x), limits[1])


def clipped_log_of_vec(vec, eps=1e-5, clip_to_zero=False):
    """
    This method takes as input a int|float or a 1D array of floats. It
    returns the log element-wise of that array, except when an element of
    the array is < eps, where eps is a positive but << 1 float. In that
    exceptional case, the method "clips the log", meaning that it returns
    log(eps) if clip_to_zero=False and 0 if clip_to_zero=True


    Parameters
    ----------
    vec : int|float|np.ndarray
    eps : float
    clip_to_zero : bool

    Returns
    -------
    np.ndarray

    """
    assert eps > 0
    vec1 = vec
    if isinstance(vec, (int, float)):
        vec1 = [vec]
    li = []
    for x in vec1:
        if x < eps:
            if clip_to_zero:
                li.append(0)
            else:
                li.append(np.log(eps))
        else:
            li.append(np.log(x))
    return np.array(li)


def positive_part(x):
    """
    This method returns max(0, x)

    Parameters
    ----------
    x : int|float

    Returns
    -------
    int|float

    """
    return max([0, x])


def positive_part_of_vec(vec):
    """
    This method takes as input a int|float or a 1D array of floats. It
    returns the array, with negative items replaced by zero

    Parameters
    ----------
    vec : int|float|np.ndarray

    Returns
    -------
    np.ndarray

    """
    vec1 = vec
    if isinstance(vec, (int, float)):
        vec1 = [vec]
    li = list(vec1)
    li_new = [max(0, x) for x in li]
    return np.array(li_new)


def max_or_zero_of_vec(vec):
    """
    This method takes as input a int|float or a 1D array of floats. It
    returns the array, with all items except the max item replaced by zero

    Parameters
    ----------
    vec : int|float|np.ndarray

    Returns
    -------
    np.ndarray

    """
    vec1 = vec
    if isinstance(vec, (int, float)):
        vec1 = [vec]
    li = list(vec1)
    (xmax, i) = max((x, i) for i, x in enumerate(li))
    vec1 = np.zeros((len(li),))
    vec1[i] = xmax
    return vec1


def fun_of_herm_arr_from_eigen_sys(fun_of_evas, evas, evec_cols,
                                   **fun_kwargs):
    """
    eigen_sys= eigensystem= (eigenvalues, eigenvectors as columns)= (evas,
    evec_cols)

    This method returns a function fun of a Hermitian matrix mat. This is
    calculated as fun(mat) = U.D.U^dag, where U=evec_cols is a unitary
    matrix with the eigenvectors of mat as columns, U^dag is the Hermitian
    conjugate of U, D= diag(fun(evas)) is a diagonal matrix whose diagonal
    is obtained by applying element-wise the function fun=fun_of_evas  to
    the 1D array of eigenvalues evas.

    Parameters
    ----------
    fun_of_evas : function
    evas : np.ndarray
    evec_cols : np.ndarray
    fun_kwargs : dict
        dict of keyword arguments that fun depends on

    Returns
    -------
    np.ndarray

    """
    # evas contains eigenvalues
    # evec_cols contains eigenvectors as columns

    umat = cp.copy(evec_cols)
    umat_H = evec_cols.conj().T
    # print(',,.', umat_H)
    # print(',,.,', umat)
    for k in range(len(evas)):
        # multiply each col of evec_cols by corresponding eigenvalue
        umat[:, k] *= fun_of_evas(evas[k], **fun_kwargs)
    # print(',,', evec_cols)
    return np.dot(umat, umat_H)


def fun_of_herm_arr(fun_of_evas, herm_arr, **fun_kwargs):
    """
    This method does the same as the method ut.fun_of_herm_arr_from_eigen(),
    except that it calculates evas and eigen_cols from the matrix herm_arr
    which it has as input.

    Parameters
    ----------
    fun_of_evas : function
        np function acting on 1d array
    herm_arr : np.ndarray
        Hermitian array
    fun_kwargs : dict
        dict of keyword args that fun depends on

    Returns
    -------
    np.ndarray

    """
    evas, evec_cols = np.linalg.eigh(herm_arr)
    return fun_of_herm_arr_from_eigen_sys(
        fun_of_evas, evas, evec_cols, **fun_kwargs)


def get_equiv_classes(li):
    """
    This method is given as input a list li of floats, some of which may be
    equal within epsilon = 1e-6. The method then returns a list of
    equivalence classes, where each equivalence class is a list of the int
    positions in li of those floats that are equal to each other within
    epsilon.

    Parameters
    ----------
    li : list[float]|np.ndarray

    Returns
    -------
    list[list[int]]

    """
    classes = []
    for k, val in enumerate(li):
        found_twin = False
        # eq_class = equivalence class
        for eq_class in classes:
            diff = val-li[eq_class[0]]
            if abs(diff) < 1e-4:
                eq_class.append(k)
                found_twin = True
                break
        if not found_twin:
            classes.append([k])
    return classes


def is_unitary_arr(umat):
    """
    Returns True iff umat is a unitary matrix

    Parameters
    ----------
    umat : np.ndarray

    Returns
    -------
    bool

    """

    return np.linalg.norm(np.dot(umat.conj().T, umat)
                          - np.eye(umat.shape[0])) < 1e-5


def is_hermitian_arr(arr):
    """
    Returns True iff arr is a Hermitian matrix.

    Returns
    -------
    bool

    """
    return np.linalg.norm(arr - arr.conj().T) < 1e-6


def is_positive_arr(arr):
    """
    This method checks that all elements of arr are > 0.

    Parameters
    ----------
    arr : np.ndarray

    Returns
    -------
    bool

    """
    out = True
    if not np.all(arr > 0):
        print('some negative neg or zero entries in ' + str(arr))
        out = False
    return out


def is_nonnegative_arr(arr):
    """
    This method checks that all elements of arr are > -1e-6.

    Parameters
    ----------
    arr : np.ndarray

    Returns
    -------
    bool

    """
    out = True
    if not np.all(arr > -1e-6):
        print('some negative entries in ' + str(arr))
        out = False
    return out


def is_prob_dist(prob_dist):
    """
    This method checks that the elements of arr define a probability
    distribution.

    Parameters
    ----------
    prob_dist : np.ndarray

    Returns
    -------
    bool

    """
    out = True
    suma = np.sum(prob_dist)
    if not np.all(prob_dist > -1e-6):
        print('some negative probs')
        out = False
    if not abs(suma - 1) < 1e-3:
        print("probs don't sum to one")
        out = False
    if not out:
        print('prob dist=\n', prob_dist)
        print('sum=', suma)
    return out


def get_entropy_from_probs(probs):
    """
    This method returns the classical entropy of the probability
    distribution probs. It checks that probs is a prob distribution.

    Parameters
    ----------
    probs : np.ndarray

    Returns
    -------
    float

    """
    assert is_prob_dist(probs)
    # print('bbbnnnnn evas', probs)
    ent = 0.0
    for val in probs:
        if val > 1e-6:
            ent += - val*np.log(val)
    return ent


def random_unitary(dim):
    """
    This method returns a random unitary matrix of size dim x dim

    Parameters
    ----------
    dim : int

    Returns
    -------
    np.ndarray

    """
    rmat = np.random.randn(dim, dim) + 1j*np.random.randn(dim, dim)
    Q, R = np.linalg.qr(rmat)
    return Q


def random_st_vec(dim):
    """
    This method returns a random complex 1D numpy array, normalized, of size
    dim.

    Parameters
    ----------
    dim : int

    Returns
    -------
    np.ndarray

    """
    st_vec = np.random.randn(dim) + 1j*np.random.randn(dim)
    st_vec /= np.linalg.norm(st_vec)
    return st_vec


def comb(n, k):
    """
    This method returns the number of combinations of k picks (with return)
    out of n possible choices, "n choose k" = n!/[(n-k)! k!]

    Parameters
    ----------
    n : int
    k : int

    Returns
    -------
    int

    """
    assert 0 <= k <= n
    ans = factorial(n) // factorial(k) // factorial(n - k)
    # print("...", ans)
    return ans


def prob_hypergeometric(x, xx, n, nn):
    """
    This method returns

    P(x | xx, n, nn) = comb(xx, x)*comb(nn-xx, n-x)/comb(nn, n)

    where
    0 <= x <= xx
    0 <= n-x <= nn-xx
    0 <= n <= nn

    This P(x | ) defines the hypergeometric distribution

    References
    ----------
    1. https://en.wikipedia.org/wiki/Hypergeometric_distribution

    Parameters
    ----------
    x : int
    xx : int
    n : int
    nn : int

    Returns
    -------
    float

    """
    if any([k < 0 for k in [xx, x, nn-xx, n-x, nn, n]]):
        return 0
    if xx < x or nn-xx < n-x or nn < n:
        return 0

    return comb(xx, x)*comb(nn-xx, n-x)/comb(nn, n)


"""
In[3]: import numpy as np
In[4]: a = np.array([[1,2], [3,4]])
In[5]: a.flatten()
Out[5]: array([1, 2, 3, 4])
In[6]: a.reshape((4,))
Out[6]: array([1, 2, 3, 4])

In[10]: a = np.kron(np.array([1, 1, 1]), np.array([1, 2]))
In[11]: a
Out[11]: array([1, 2, 1, 2, 1, 2])
In[12]: a.reshape((3, 2))
Out[12]:
array([[1, 2],
       [1, 2],
       [1, 2]])

"""

if __name__ == "__main__":
    def main():
        fprod = scalar_prod([5, 6.1, 3])
        print('scalar_prod=', fprod)

        h = np.array([[1, 1], [1, -1]])
        h2 = kron_prod([h, h])
        print('h2=\n', h2)
        h3 = kron_prod([h, h, h])
        print('h3=\n', h3)

        v = np.array([2, 3.1])
        me = mat_elem(v, h, v)
        print('me=', me)

        v = np.array([2, 1e-6, -1e-6])
        print('v=', v, '\nlog(v)=', clipped_log_of_vec(v))

        print('exp_h2=\n', fun_of_herm_arr(np.exp, h2))

        li = [.3, .1, .2, .1, .3, .3]
        print('li=', li)
        print('equiv classes of li=', get_equiv_classes(li))

        probs = np.array(li)
        probs /= np.sum(probs)
        print('entropy=', get_entropy_from_probs(probs))

        rmat = random_unitary(3)
        print('random unitary\n', rmat)

        print('hypergeo=', prob_hypergeometric(1, 2, 3, 4))

    main()


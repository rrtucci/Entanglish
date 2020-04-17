from entanglish.DenMat import *
from entanglish.DenMatPertTheory import *
from entanglish.MaxEntangState import *
import numpy as np
import itertools as it


class EntangCase:
    """
    This is an abstract class meant to be overridden. It is the parent class
    of classes PureStEnt and SquashedEnt, which calculate quantum
    entanglement for pure and mixed states using various methods. This class
    contains methods useful to all of its children. For example, it contains
    methods that construct an entanglement profile data structure and print
    it.

    Attributes
    ----------
    approx : str
        approx used to calculate the (natural) log of a density matrix that
        appears in the definition of the von Neumann entropy. This parameter
        can be either 'eigen', 'pade', 'pert'.
    num_bstrap_steps : int
        number bootstrap steps used in perturbation theory. Only used if
        approx = 'pert'
    num_row_axes : int
        number of row axes, same as number of qudits, equal to len(
        row_shape) of a DenMat
    verbose : bool

    """

    def __init__(self, num_row_axes, approx='eigen', num_bstrap_steps=1,
                 verbose=False):
        """
        Constructor

        Parameters
        ----------
        num_row_axes : int
        approx : str
        num_bstrap_steps : int
        verbose : bool

        Returns
        -------


        """
        self.num_row_axes = num_row_axes
        self.approx = approx
        assert approx in ['eigen', 'pade', 'pert']
        self.num_bstrap_steps = num_bstrap_steps
        assert num_bstrap_steps > 0
        self.verbose = verbose

    def mirror(self, x):
        """
        Given an element of the list range(self.num_row_axes), this method
        returns the mirror image of x with respect to the center of the list.

        Parameters
        ----------
        x : int

        Returns
        -------
        int

        """
        assert 0 <= x < self.num_row_axes
        return self.num_row_axes - 1 - x

    def mirror_many(self, xx):
        """
        This applies the method self.mirror() to each element of a list,
        set, tuple

        Parameters
        ----------
        xx : tuple[int]|set[int]|list[int]

        Returns
        -------
        tuple[int]|set[int]|list[int]

        """
        if isinstance(xx, list):
            return list([self.mirror(k) for k in xx])
        elif isinstance(xx, set):
            return set([self.mirror(k) for k in xx])
        elif isinstance(xx, tuple):
            return tuple([self.mirror(k) for k in xx])
        else:
            assert False

    def get_entang(self, axes_subset):
        """
        This is an abstract method that should be overridden by the children
        of the class.

        Parameters
        ----------
        axes_subset : set[int]
            the entanglement is calculated for parts axes_subset and its
            complement

        Returns
        -------
        float

        """
        assert False

    @staticmethod
    def check_max_entang_st(st):
        """
        This method checks that the object st of class MaxEntangState does
        indeed carry maximal entanglement. The entanglement is calculated 3
        different ways (von Neumann entropy of 2 partial traces of density
        matrix, and from known value) and the 3 values are checked to agree.

        Parameters
        ----------
        st : MaxEntangState

        Returns
        -------
        None

        """
        dm = DenMat(st.num_rows, st.row_shape)
        st_vec = st.get_st_vec()
        dm.set_arr_from_st_vec(st_vec)
        ent1 = dm.get_partial_tr(set(st.y_axes)).get_entropy()
        ent2 = dm.get_partial_tr(set(st.x_axes)).get_entropy()
        ent3 = np.log(st.num_vals_min)
        assert abs(ent1 - ent2) < 1e-6 and abs(ent1 - ent3) < 1e-6, \
            str(ent1) + ', ' + str(ent2) + ', ' + str(ent3)

    @staticmethod
    def get_max_entag(row_shape, x_axes, y_axes):
        """
        This method returns the maximum possible entanglement with parts
        x_axes, y_axes. x_axes, y_axes give a bi-partition of range( len(
        row_shape)). If num_vals_min = min(num_vals_x, num_vals_y),
        then max-entang is the log of num_vals_min.

        Parameters
        ----------
        row_shape : tuple[int]
        x_axes : list{int]
        y_axes : list{int]

        Returns
        -------
        float

        """
        assert len(row_shape) == len(set(x_axes).union(set(y_axes)))
        num_axes_x = len(x_axes)
        num_axes_y = len(y_axes)
        row_shape_x = [row_shape[k] for k in x_axes]
        row_shape_y = [row_shape[k] for k in y_axes]
        num_vals_x = np.product(np.array(row_shape_x))
        num_vals_y = np.product(np.array(row_shape_y))
        num_vals_min = min(num_vals_x, num_vals_y)
        return np.log(num_vals_min)

    def get_entang_profile(self):
        """
        This method constructs a dictionary that we call an entanglement
        profile. Given a state with num_row_axes qudits, this method
        calculates a (bipartite) entanglement for each possible bi-partition
        of range( num_row_axes). By a bi-partition we mean two nonempty
        disjoint subsets whose union is range(num_row_axes). An entanglement
        profile is a dictionary mapping bi-partition half-size to a
        dictionary that maps each bi-partition of that half-size to its
        entanglement.

        Returns
        -------
        dict[int, dict[tuple[int], float]]

        """
        # 5 axes, max_comb_len=2= 5//2
        # 4 axes, max_comb_len=2= 4//2
        max_comb_len = self.num_row_axes//2
        all_axes = list(range(0, self.num_row_axes))
        entang_profile = {}
        for comb_len in range(1, max_comb_len+1):
            entang_profile[comb_len] = {}
            for traced_axes in it.combinations(all_axes, comb_len):
                entang_profile[comb_len][tuple(sorted(traced_axes))] = \
                    self.get_entang(set(traced_axes))
        return entang_profile

    def print_entang_profiles(self, profile_list, row_shape=None):
        """
        This method takes as input a list of entanglement profiles. Each
        element of that list can be generated with the method
        get_entang_profile(). The method prints, side by side, all the
        entanglement profiles in the input profile list. If a row_shape is
        provided, the last column of the profile is the maximum entanglement.

        Parameters
        ----------
        profile_list : list[dict[int, dict[tuple[int], float]]]
        row_shape : tuple[int]

        Returns
        -------
        None

        """
        # 5 axes, max_comb_len=2= 5//2
        # 4 axes, max_comb_len=2= 4//2
        max_comb_len = self.num_row_axes//2
        assert all(len(profile_list[k]) == max_comb_len
                   for k in range(len(profile_list)))
        all_lines = ''
        for comb_len in range(1, max_comb_len+1):
            all_lines += 'bi-partition half-size=' + str(comb_len) + '\n'
            for traced_axes in profile_list[0][comb_len].keys():
                comb = list(traced_axes)
                comb_c = [k for k in range(self.num_row_axes)
                              if k not in traced_axes]
                line = '(' + str(comb)[1:-1] \
                       + ' | ' \
                       + str(comb_c)[1:-1] + ")"
                line += ' :\t'
                for pf_index, pf in enumerate(profile_list):
                    line += "{:8.5f}".format(pf[comb_len][traced_axes])
                    if pf_index != len(profile_list)-1:
                        line += ", "
                    else:
                        if row_shape:
                            line += ', max-entang=' + \
                                "{:8.5f}".format(EntangCase.get_max_entag(
                                    row_shape, comb, comb_c)) + '\n'
                        else:
                            line += '\n'
                all_lines += line
        print(all_lines)

    def sqrt(self, dm, approx=None):
        """
        This method is a simple wrapper for dm.sqrt() so that all usages
        inside the class utilize approx=self.approx if approx!=None.

        Parameters
        ----------
        dm : DenMat
        approx : str

        Returns
        -------
        DenMat

        """
        if approx is None:
            approx = self.approx
        dm_out = None
        if approx in ['eigen', 'pade']:
            dm_out = dm.sqrt(approx=approx)
        elif approx == 'pert':
            esys = DenMatPertTheory.do_bstrap_with_separable_dm0(
                dm, self.num_bstrap_steps)
            dm_out = DenMat.get_fun_of_dm_from_eigen_sys(
                dm.num_rows, dm.row_shape, esys, np.sqrt)
        else:
            assert False
        return dm_out

    def exp(self, dm, approx=None):
        """
        This method is a simple wrapper for dm.exp() so that all usages
        inside the class utilize approx=self.approx if approx!=None.

        Parameters
        ----------
        dm : DenMat
        approx : str

        Returns
        -------
        DenMat

        """
        if approx is None:
            approx = self.approx
        dm_out = None
        if approx in ['eigen', 'pade']:
            dm_out = dm.exp(approx)
        elif approx == 'pert':
            esys = DenMatPertTheory.do_bstrap_with_separable_dm0(
                dm, self.num_bstrap_steps)
            dm_out = DenMat.get_fun_of_dm_from_eigen_sys(
                dm.num_rows, dm.row_shape, esys, np.exp)
        else:
            assert False
        return dm_out

    def log(self, dm, approx=None):
        """
        This method is a simple wrapper for dm.log() so that all usages
        inside the class utilize approx=self.approx if approx!=None.

        Parameters
        ----------
        dm : DenMat
        approx : str|None

        Returns
        -------
        DenMat

        """
        if approx is None:
            approx = self.approx
        dm_out = None
        if approx in ['eigen', 'pade']:
            dm_out = dm.log(approx=approx)
        elif approx == 'pert':
            esys = DenMatPertTheory.do_bstrap_with_separable_dm0(
                dm, self.num_bstrap_steps)
            dm_out = DenMat.get_fun_of_dm_from_eigen_sys(
                dm.num_rows, dm.row_shape, esys, np.log)
        else:
            assert False
        return dm_out


if __name__ == "__main__":
    from entanglish.DenMat import *
    from entanglish.SymNupState import *
    from entanglish.PureStEnt import *

    def main():
        num_qbits = 5
        num_up = 2
        dm = DenMat(1 << num_qbits, tuple([2]*num_qbits))
        st = SymNupState(num_up, num_qbits)
        st_vec = st.get_st_vec()
        dm.set_arr_from_st_vec(st_vec)
        ecase = PureStEnt(dm, 'eigen')
        pf = ecase.get_entang_profile()
        # print(',,...', pf)
        ecase.print_entang_profiles([pf, pf], dm.row_shape)

    main()

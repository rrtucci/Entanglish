import itertools as it


class Entang:
    """
    This is an abstract class meant to be overridden. It is the parent class
    of classes EntangPureSt and SquashedEnts, which calculate quantum
    entanglement for pure and mixed states using various methods. This class
    contains methods useful to all of its children. For example, it contains
    methods that construct an entanglement profile data structure and print
    it.

    Attributes
    ----------
    num_row_axes : int
        number of row axes, same as number of qudits, equal to len(
        row_shape) of a DenMat

    """

    def __init__(self, num_row_axes):
        """
        Constructor

        Parameters
        ----------
        num_row_axes : int

        Returns
        -------


        """
        self.num_row_axes = num_row_axes

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

    def get_entang(self, traced_axes_set):
        """
        This is an abstract method that should be overridden by the children
        of the class.

        Parameters
        ----------
        traced_axes_set : set[int]

        Returns
        -------
        float

        """
        assert False

    def get_entang_profile(self):
        """
        This method constructs a dictionary that we call an entanglement
        profile. Given a state with num_row_axes qudits, this method calculates
        a (bipartite) entanglement for each possible partition of range(
        num_row_axes). By a partition we mean two nonempty disjoint subsets
        whose union is range(num_row_axes). An entanglement profile is a
        dictionary that maps each of all possible partitions to its
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
                entang_profile[comb_len][tuple(traced_axes)] = \
                    self.get_entang(set(traced_axes))
        return entang_profile

    def print_entang_profiles(self, profile_list, print_mir_axes=False):
        """
        This method takes as input a list of entanglement profiles. Each
        element of that list can be generated with the method
        get_entang_profile(). The method prints, side by side, all the
        entanglement profiles in the input list.

        Parameters
        ----------
        profile_list : list[dict[int, dict[tuple[int], float]]]
        print_mir_axes : bool
            print mirror axes. If this is changed to True from its default
            of False, the method applies a mirror_many() transformation to
            the partition label of each entanglement value of the
            entanglement profile.

        Returns
        -------
        str

        """
        # 5 axes, max_comb_len=2= 5//2
        # 4 axes, max_comb_len=2= 4//2
        max_comb_len = self.num_row_axes//2
        assert all(len(profile_list[k]) == max_comb_len
                   for k in range(len(profile_list)))
        all_lines = ''
        for comb_len in range(1, max_comb_len+1):
            all_lines += str(comb_len) + '\n'
            for traced_axes in profile_list[0][comb_len].keys():
                comb_set = set(traced_axes)
                comb_set_c = {k for k in range(self.num_row_axes)
                              if k not in traced_axes}
                if not print_mir_axes:
                    line = str(comb_set) \
                           + ' | ' \
                           + str(comb_set_c)
                else:
                    line = str(self.mirror_many(comb_set))\
                           + ' | ' \
                           + str(self.mirror_many(comb_set_c))
                line += ':\t'
                for pf_index, pf in enumerate(profile_list):
                    line += str(pf[comb_len][traced_axes])
                    if pf_index != len(profile_list)-1:
                        line += ", "
                    else:
                        line += '\n'
                all_lines += line
        print(all_lines)
        return all_lines

if __name__ == "__main__":
    from entanglish.DenMat import *
    from entanglish.SymNupState import *
    from entanglish.EntangPureSt import *

    def main():
        num_bits = 5
        num_up = 2
        dm = DenMat(1 << num_bits, tuple([2]*num_bits))
        st = SymNupState(num_up, num_bits)
        st_vec = st.get_st_vec()
        dm.set_arr_from_st_vec(st_vec)
        entang_pure = EntangPureSt(dm, 'exact-eigen')
        pf = entang_pure.get_entang_profile()
        # print(',,...', pf)
        entang_pure.print_entang_profiles([pf, pf], print_mir_axes=True)

    main()

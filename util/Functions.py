from torch_scatter import scatter_mean, scatter_min, scatter_max, scatter_sum, scatter_std

from util.Types import *
from functools import partial, update_wrapper
import numpy as np
import copy
import collections


def get_from_nested_dict(dictionary: Dict[Any, Any], list_of_keys: List[Any],
                         raise_error: bool = False,
                         default_return: Optional[Any] = None) -> Any:
    """
    Utility function to traverse through a nested dictionary. For a given dict d and a list of keys [k1, k2, k3], this
    will return d.get(k1).get(k2).get(k3) if it exists, and default_return otherwise
    Args:
        dictionary: The dictionary to search through
        list_of_keys: List of keys in the order to traverse the dictionary by
        raise_error: Raises an error if any of the keys is missing in its respective subdict. If false, returns the
        default_return instead
        default_return: The thing to return if the dictionary does not contain the next key at some level

    Returns:

    """
    current_dict = dictionary
    for key in list_of_keys:
        if isinstance(current_dict, dict):  # still traversing dictionaries
            current_dict = current_dict.get(key, None)
        elif current_dict is None:  # key of sub-dictionary not found
            if raise_error:
                raise ValueError("Dict '{}' does not contain list_of_keys '{}'".format(dictionary, list_of_keys))
            else:
                return default_return
    return current_dict  # bottom level reached


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def joint_sort(main_array, *args, reverse=False) -> tuple:
    """
    Jointly sorts np arrays a and b with same length and possibly different dimensions
    Args:
        main_array: Some sortable array. Will be used to sort the other arrays by
        reverse: Whether to flip an array or not
        args: Additional arrays to be sorted alongside a.

    Returns:
        The sorted array a and the array b sorted in the same way

    """
    if isinstance(main_array, list):
        main_array = np.array(main_array)

    if reverse:
        positions = np.squeeze(-main_array).argsort()
    else:
        positions = np.squeeze(main_array).argsort()

    additional_arrays = []
    for arr in args:
        if isinstance(arr, list):
            arr = np.array(arr)
        additional_arrays.append(arr[positions])
    return main_array[positions], *additional_arrays


def list_of_dicts_from_dict_of_lists(nested_dict: ConfigDict) -> List[ConfigDict]:
    """
    Takes a nested dictionary where some values may be lists and returns a new dictionary for each row-wise combination
    of elements for this list. These new dictionaries are then put in an outer list.
    E.g., a dictionary
    a:
      b: 1
      c: [2,3]
    d:
      e: [4,5]
      f: 6
    g: [7,8]

    will be turned into a list
    [a:
      b: 1
      c: 2
    d:
      e: 4
      f: 6
    g: 7,

    a:
      b: 1
      c: 3
    d:
      e: 5
      f: 6
    g: 8]
    Args:
        nested_dict: The nested dictionary to transform. May contain list elements, in which case all lists must be
          of the same length

    Returns: A new dictionary that is a merged version of the two provided ones

    """

    def list_length_in_dict(checked_dictionary: ConfigDict) -> Optional[int]:
        length = 1
        for name, value in checked_dictionary.items():
            if isinstance(value, dict):  # recurse
                sublength = list_length_in_dict(checked_dictionary=value)
            elif isinstance(value, list):  # is a list, so take this length
                sublength = len(value)
            else:
                sublength = 1
            if sublength > 1:
                if length == 1:
                    length = sublength
                else:
                    assert length == sublength, f"Need all list entries to have common length" \
                                                f" '{length}'. Got length '{sublength}' for key '{name}'"
        return length

    def take_nth_elements(dictionary, n: int):
        for name, value in dictionary.items():
            if isinstance(value, dict):  # recurse
                dictionary[name] = take_nth_elements(value, n=n)
            elif isinstance(value, list):  # is a list, so split evenly
                dictionary[name] = value[n]
        return dictionary

    list_length = list_length_in_dict(copy.deepcopy(nested_dict))
    list_of_dicts = []
    for position in range(list_length):
        configuration = take_nth_elements(dictionary=copy.deepcopy(nested_dict), n=position)
        list_of_dicts.append(configuration)
    return list_of_dicts


def flatten_dict_to_tuple_keys(d: collections.MutableMapping):
    flat_dict = {}
    for k, v in d.items():
        if isinstance(v, collections.MutableMapping):
            sub_dict = flatten_dict_to_tuple_keys(v)
            flat_dict.update({(k, *sk): sv for sk, sv in sub_dict.items()})

        elif isinstance(v, collections.MutableSequence):
            flat_dict[(k,)] = v

    return flat_dict


def merge_nested_dictionaries(destination: dict, source: dict) -> dict:
    """
    does a deep merge of the given dictionaries. The destination dictionary will be overwritten by the source one in
    case of conflicting keys
    Args:
        destination:
        source:

    Returns: A new dictionary that is a merged version of the two provided ones

    """

    def _merge_dictionaries(dict1, dict2):
        for key in dict2:
            if key in dict1:
                if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                    _merge_dictionaries(dict1[key], dict2[key])
                elif dict1[key] == dict2[key]:
                    pass  # same leaf value
                else:
                    dict1[key] = dict2[key]
            else:
                dict1[key] = dict2[key]
        return dict1

    return _merge_dictionaries(copy.deepcopy(destination), copy.deepcopy(source))


def remove_key(dictionary: dict, key_to_remove: str, create_copy: bool = True) -> dict:
    """
    Removes a key from the given dictionary if existing. Optionally creates a copy beforehand to not affect other
    references to the same dictionary
    Args:
        dictionary:
        key_to_remove:
        create_copy: Whether to create a copy before removing the key or not

    Returns: (A potential copy of) the same dictionary, but without the key that was removed

    """
    import copy
    if key_to_remove in dictionary.keys():
        if create_copy:
            dictionary = copy.deepcopy(dictionary)
        del dictionary[key_to_remove]
    return dictionary


def prefix_keys(dictionary: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {prefix + "/" + k: v for k, v in dictionary.items()}


def add_to_dictionary(dictionary: ValueDict, new_scalars: ValueDict) -> ValueDict:
    for k, v in new_scalars.items():
        if k not in dictionary:
            dictionary[k] = []
        if isinstance(v, list) or (isinstance(v, np.ndarray) and v.ndim == 1):
            dictionary[k] = dictionary[k] + v
        else:
            dictionary[k].append(v)
    return dictionary


def get_aggregation_function(name: str) -> callable:
    if name == "mean":
        aggregation_function = scatter_mean
    elif name == "min":
        aggregation_function = lambda *args, **kwargs: scatter_min(*args, **kwargs)[0]
    elif name == "max":
        aggregation_function = lambda *args, **kwargs: scatter_max(*args, **kwargs)[0]
    elif name == "sum":
        aggregation_function = scatter_sum
    elif name == "std":
        aggregation_function = scatter_std
    else:
        raise ValueError(f"Unknown aggregation function '{name}'")  # todo add other options
    return aggregation_function

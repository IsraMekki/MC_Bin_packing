def copy_dict_of_dict(dict_a):
    dict_b = dict(dict_a)
    for k in dict_b.keys():
        dict_b[k] = dict(dict_a[k])
    return dict_b
'''Manipulate strings'''


def underscore2camelcase(s):
    assert s == s.lower(), 'Invalid underscore string: no upper case character is allowed, in "{}"'.format(s)
    assert all([x.isdigit() or x.isalpha() or x == '_' for x in s]),\
        'Invalid underscore, all character must be letters or numbers or underscore'
    terms = s.split('_')
    for x in terms:
        assert x, 'Invalid underscore string: no consecutive _ is allowed, in "{}"'.format(s)
        assert x[0].upper() != x[0], \
            'Invalid underscore string: phrases must start with a character, in "{}'.format(s)

    return ''.join([x[0].upper() + x[1:] for x in terms if x])


def camelcase2underscore(s):
    assert s[0].isupper(), 'Invalid camel case, first character must be upper case, in "{}"'.format(s)
    assert all([x.isdigit() or x.isalpha() for x in s]),\
        'Invalid camel case, all character must be letters or numbers'

    out = s[0].lower()
    for x in s[1:]:
        if x.lower() != x:
            out += '_' + x.lower()
        else:
            out += x
    return out

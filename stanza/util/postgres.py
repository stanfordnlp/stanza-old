"""
Utilities to use when interfacing with Postgres.
- These utilities support the workflow wherein you store annoated
  sentences in a database.
"""
__author__ = 'arunchaganty'
import os
import stanza
import requests
import logging

def unescape_sql(inp):
    """
    :param inp: an input string to be unescaped
    :return: return the unescaped version of the string.
    """
    if inp.startswith('"') and inp.endswith('"'):
        inp = inp[1:-1]
    return inp.replace('""','"').replace('\\\\','\\')

def parse_psql_array(inp):
    """
    :param inp: a string encoding an array
    :return: the array of elements as represented by the input
    """
    inp = unescape_sql(inp)
    # Strip '{' and '}'
    if inp.startswith("{") and inp.endswith("}"):
        inp = inp[1:-1]

    lst = []
    elem = ""
    in_quotes, escaped = False, False

    for ch in inp:
        if escaped:
            elem += ch
            escaped = False
        elif ch == '"':
            in_quotes = not in_quotes
            escaped = False
        elif ch == '\\':
            escaped = True
        else:
            if in_quotes:
                elem += ch
            elif ch == ',':
                lst.append(elem)
                elem = ""
            else:
                elem += ch
            escaped = False
    if len(elem) > 0:
        lst.append(elem)
    return lst

def test_parse_psql_array():
    """
    test case for parse_psql_array
    """
    inp = '{Bond,was,set,at,$,"1,500",each,.}'
    lst = ["Bond", "was", "set", "at", "$", "1,500", "each","."]
    lst_ = parse_psql_array(inp)
    assert all([x == y for (x,y) in zip(lst, lst_)])

def escape_sql(inp):
    """
    :param inp: an input string to be escaped
    :return: return the escaped version of the string.
    """
    return '"' + inp.replace('"','""').replace('\\','\\\\') + '"'

def to_psql_array(inp):
    """
    :param inp: an array to be encoded.
    :return: a string encoding the array
    """
    return "{" + ",".join(map(escape_sql, inp)) + "}"

def test_to_psql_array():
    """
    Test for to_psql_array
    """
    inp = ["Bond", "was", "set", "at", "$", "1,500", "each","."]
    out = '{"Bond","was","set","at","$","1,500","each","."}'
    out_ = to_psql_array(inp)
    assert out == out_


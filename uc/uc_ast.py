import sys


def represent_node(obj, indent):
    def _repr(obj, indent, printed_set):
        """
        Get the representation of an object, with dedicated pprint-like format for lists.
        """
        if isinstance(obj, list):
            indent += 1
            sep = ",\n" + (" " * indent)
            final_sep = ",\n" + (" " * (indent - 1))
            return (
                "["
                + (sep.join((_repr(e, indent, printed_set) for e in obj)))
                + final_sep
                + "]"
            )
        elif isinstance(obj, Node):
            if obj in printed_set:
                return ""
            else:
                printed_set.add(obj)
            result = obj.__class__.__name__ + "("
            indent += len(obj.__class__.__name__) + 1
            attrs = []
            for name in obj.__slots__[:-1]:
                if name == "bind":
                    continue
                value = getattr(obj, name)
                value_str = _repr(value, indent + len(name) + 1, printed_set)
                attrs.append(name + "=" + value_str)
            sep = ",\n" + (" " * indent)
            final_sep = ",\n" + (" " * (indent - 1))
            result += sep.join(attrs)
            result += ")"
            return result
        elif isinstance(obj, str):
            return obj
        else:
            return ""

    # avoid infinite recursion with printed_set
    printed_set = set()
    return _repr(obj, indent, printed_set)


class Node:
    """Abstract base class for AST nodes."""

    __slots__ = "coord"
    attr_names = ()

    def __init__(self, coord=None):
        self.coord = coord

    def __repr__(self):
        """Generates a python representation of the current node"""
        return represent_node(self, 0)

    def children(self):
        """A sequence of all children that are Nodes"""
        pass

    def show(
        self,
        buf=sys.stdout,
        offset=0,
        attrnames=False,
        nodenames=False,
        showcoord=False,
        _my_node_name=None,
    ):
        """Pretty print the Node and all its attributes and children (recursively) to a buffer.
        buf:
            Open IO buffer into which the Node is printed.
        offset:
            Initial offset (amount of leading spaces)
        attrnames:
            True if you want to see the attribute names in name=value pairs. False to only see the values.
        nodenames:
            True if you want to see the actual node names within their parents.
        showcoord:
            Do you want the coordinates of each Node to be displayed.
        """
        lead = " " * offset
        if nodenames and _my_node_name is not None:
            buf.write(lead + self.__class__.__name__ + " <" + _my_node_name + ">: ")
            inner_offset = len(self.__class__.__name__ + " <" + _my_node_name + ">: ")
        else:
            buf.write(lead + self.__class__.__name__ + ":")
            inner_offset = len(self.__class__.__name__ + ":")

        if self.attr_names:
            if attrnames:
                nvlist = [
                    (
                        n,
                        represent_node(
                            getattr(self, n), offset + inner_offset + 1 + len(n) + 1
                        ),
                    )
                    for n in self.attr_names
                    if getattr(self, n) is not None
                ]
                attrstr = ", ".join("%s=%s" % nv for nv in nvlist)
            else:
                vlist = [getattr(self, n) for n in self.attr_names]
                attrstr = ", ".join(
                    represent_node(v, offset + inner_offset + 1) for v in vlist
                )
            buf.write(" " + attrstr)

        if showcoord:
            if self.coord and self.coord.line != 0:
                buf.write(" %s" % self.coord)
        buf.write("\n")

        for (child_name, child) in self.children():
            child.show(buf, offset + 4, attrnames, nodenames, showcoord, child_name)


class Program(Node):
    __slots__ = ("gdecls", "coord", "text")

    def __init__(self, gdecls, coord=None):
        self.gdecls = gdecls
        self.coord = coord

        self.text = None

    def children(self):
        nodelist = []
        for i, child in enumerate(self.gdecls or []):
            nodelist.append(("gdecls[%d]" % i, child))
        return tuple(nodelist)

    attr_names = ()


class GlobalDecl(Node):
    __slots__ = ("decls", "coord", "scope")

    def __init__(self, decls, coord=None):
        self.decls = decls
        self.coord = coord

        self.scope = None

    def children(self):
        nodelist = []
        for i, child in enumerate(self.decls or []):
            nodelist.append(("decls[%d]" % i, child))
        return tuple(nodelist)

    attr_names = ()


class Decl(Node):
    __slots__ = ("declname", "type", "init", "coord", "scope", "uc_type")

    def __init__(self, declname, type, init, coord=None):
        self.declname = declname
        self.type = type
        self.init = init
        self.coord = coord

        self.scope = None
        self.uc_type = None

    def children(self):
        nodelist = []
        if self.type is not None:
            nodelist.append(("type", self.type))
        if self.init is not None:
            nodelist.append(("init", self.init))
        return tuple(nodelist)

    attr_names = ("declname",)


class VarDecl(Node):
    __slots__ = ("declname", "type", "init", "coord", "scope", "uc_type")

    def __init__(self, declname, type, init, coord=None):
        self.declname = declname
        self.type = type
        self.init = init
        self.coord = coord

        self.scope = None
        self.uc_type = None

    def children(self):
        nodelist = []
        if self.type is not None:
            nodelist.append(("type", self.type))
        if self.init is not None:
            nodelist.append(("init", self.init))
        return tuple(nodelist)

    attr_names = ()


class ArrayDecl(Node):
    __slots__ = ("declname", "type", "init", "coord", "scope", "uc_type")

    def __init__(self, declname, type, init, coord=None):
        self.declname = declname
        self.type = type
        self.init = init
        self.coord = coord

        self.uc_type = None

    def children(self):
        nodelist = []
        if self.type is not None:
            nodelist.append(("type", self.type))
        if self.init is not None:
            nodelist.append(("init", self.init))

        return tuple(nodelist)

    attr_names = ()


class FuncDef(Node):
    __slots__ = (
        "type",
        "decl",
        "declList",
        "compound",
        "coord",
        "scope",
        "uc_type",
        "cfg",
    )

    def __init__(self, type, decl, declList, compound, coord=None):
        self.type = type
        self.decl = decl
        self.declList = declList
        self.compound = compound
        self.coord = coord

        self.scope = None
        self.uc_type = None

        self.cfg = None

    def children(self):
        nodelist = []
        if self.type is not None:
            nodelist.append(("type", self.type))
        if self.decl is not None:
            nodelist.append(("decl", self.decl))
        if self.declList is not None:
            nodelist.append(("declList", self.declList))
        if self.compound is not None:
            nodelist.append(("compound", self.compound))
        return tuple(nodelist)

    attr_names = ()


class FuncDecl(Node):
    __slots__ = ("declname", "type", "init", "coord", "scope", "uc_type")

    def __init__(self, declname, type, init, coord=None):
        self.declname = declname
        self.type = type
        self.init = init
        self.coord = coord

        self.scope = None
        self.uc_type = None

    def children(self):
        nodelist = []
        if self.init is not None:
            nodelist.append(("init", self.init))
        if self.type is not None:
            nodelist.append(("type", self.type))

        return tuple(nodelist)

    attr_names = ()


class ParamList(Node):
    __slots__ = ("params", "coord", "ref")

    def __init__(self, params, coord=None):
        self.params = params
        self.coord = coord

        self.ref = None

    def children(self):
        nodelist = []
        for i, child in enumerate(self.params or []):
            nodelist.append(("params[%d]" % i, child))
        return tuple(nodelist)

    attr_names = ()


class InitList(Node):
    __slots__ = ("inits", "coord", "ref", "uc_type")

    def __init__(self, inits, coord=None):
        self.inits = inits
        self.coord = coord

        self.ref = None
        self.uc_type = None

    def children(self):
        nodelist = []
        for i, child in enumerate(self.inits or []):
            nodelist.append(("init[%d]" % i, child))
        return tuple(nodelist)

    attr_names = ()


class ExprList(Node):
    __slots__ = ("exprs", "coord")

    def __init__(self, exprs, coord=None):
        self.exprs = exprs
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.exprs or []):
            nodelist.append(("exprs[%d]" % i, child))
        return tuple(nodelist)

    attr_names = ()


class DeclList(Node):
    __slots__ = ("decls", "coord")

    def __init__(self, decls, coord=None):
        self.decls = decls
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.decls or []):
            nodelist.append(("decls[%d]" % i, child))
        return tuple(nodelist)

    attr_names = ()


class BinaryOp(Node):
    __slots__ = ("op", "lvalue", "rvalue", "coord", "uc_type", "gen_location")

    def __init__(self, op, left, right, coord=None):
        self.op = op
        self.lvalue = left
        self.rvalue = right
        self.coord = coord

        self.uc_type = None
        self.gen_location = None

    def children(self):
        nodelist = []
        if self.lvalue is not None:
            nodelist.append(("lvalue", self.lvalue))
        if self.rvalue is not None:
            nodelist.append(("rvalue", self.rvalue))
        return tuple(nodelist)

    attr_names = ("op",)


class UnaryOp(Node):
    __slots__ = ("op", "unexp", "coord", "uc_type", "gen_location")

    def __init__(self, op, unexp, coord=None):
        self.op = op
        self.unexp = unexp
        self.coord = coord

        self.uc_type = None
        self.gen_location = None

    def children(self):
        nodelist = []
        if self.unexp is not None:
            nodelist.append(("unexp", self.unexp))
        return tuple(nodelist)

    attr_names = ("op",)


class Assignment(Node):
    __slots__ = ("op", "lvalue", "rvalue", "coord")

    def __init__(self, op, lvalue, rvalue, coord=None):
        self.op = op
        self.lvalue = lvalue
        self.rvalue = rvalue
        self.coord = coord

    def children(self):
        nodelist = []
        if self.lvalue is not None:
            nodelist.append(("lvalue", self.lvalue))
        if self.rvalue is not None:
            nodelist.append(("rvalue", self.rvalue))
        return tuple(nodelist)

    attr_names = ("op",)


class ArrayRef(Node):
    __slots__ = ("name", "position", "coord", "ref", "uc_type", "gen_location")

    def __init__(self, name, position, coord=None):
        self.name = name
        self.position = position
        self.coord = coord

        self.ref = None
        self.uc_type = None
        self.gen_location = None

    def children(self):
        nodelist = []
        if self.name is not None:
            nodelist.append(("name", self.name))
        if self.position is not None:
            nodelist.append(("position", self.position))
        return tuple(nodelist)

    attr_names = ()


class FuncCall(Node):
    __slots__ = ("name", "args", "coord", "ref", "uc_type", "gen_location")

    def __init__(self, name, args, coord=None):
        self.name = name
        self.args = args
        self.coord = coord

        self.ref = None
        self.uc_type = None

        self.gen_location = None

    def children(self):
        nodelist = []
        if self.name is not None:
            nodelist.append(("name", self.name))
        if self.args is not None:
            nodelist.append(("args", self.args))
        return tuple(nodelist)

    attr_names = ()


class Constant(Node):
    __slots__ = ("type", "value", "coord", "uc_type", "gen_location")

    def __init__(self, type, value, coord=None):
        self.type = type
        self.value = value
        self.coord = coord

        self.uc_type = None

        self.gen_location = None

    def children(self):
        return ()

    attr_names = (
        "type",
        "value",
    )


class ID(Node):
    __slots__ = ("name", "coord", "scope", "uc_type", "gen_location")

    def __init__(self, name, coord=None):
        self.name = name
        self.coord = coord

        self.scope = None
        self.uc_type = None
        self.gen_location = None

    def children(self):
        return ()

    attr_names = ("name",)


class Type(Node):
    __slots__ = ("name", "coord", "uc_type")

    def __init__(self, name, coord=None):
        self.name = name
        self.coord = coord

        self.uc_type = None

    def children(self):
        return ()

    attr_names = ("name",)


class If(Node):
    __slots__ = ("exp", "istats", "estats", "coord", "scope")

    def __init__(self, exp, istats, estats, coord=None):
        self.exp = exp
        self.istats = istats
        self.estats = estats
        self.coord = coord

        self.scope = None

    def children(self):
        nodelist = []
        if self.exp is not None:
            nodelist.append(("exp", self.exp))
        if self.istats is not None:
            nodelist.append(("istats", self.istats))
        if self.estats is not None:
            nodelist.append(("estats", self.estats))
        return tuple(nodelist)


class While(Node):
    __slots__ = ("exp", "stats", "coord", "scope")

    def __init__(self, exp, stats, coord=None):
        self.exp = exp
        self.stats = stats
        self.coord = coord

        self.scope = None

    def children(self):
        nodelist = []
        if self.exp is not None:
            nodelist.append(("exp", self.exp))
        if self.stats is not None:
            nodelist.append(("stats", self.stats))
        return tuple(nodelist)


class For(Node):
    __slots__ = ("left", "mid", "right", "stat", "coord", "scope")

    def __init__(self, left, mid, right, stat, coord=None):
        self.left = left
        self.mid = mid
        self.right = right
        self.stat = stat
        self.coord = coord

        self.scope = None

    def children(self):
        nodelist = []
        if self.left is not None:
            nodelist.append(("left", self.left))
        if self.mid is not None:
            nodelist.append(("mid", self.mid))
        if self.right is not None:
            nodelist.append(("right", self.right))
        if self.stat is not None:
            nodelist.append(("stat", self.stat))
        return tuple(nodelist)

    attr_names = ()


class Compound(Node):
    __slots__ = ("decls", "stats", "coord", "scope", "doNotCreateScope")

    def __init__(self, decls, stats, coord=None):
        self.decls = decls
        self.stats = stats
        self.coord = coord

        self.scope = None
        self.doNotCreateScope = None

    def children(self):
        nodelist = []
        for i, child in enumerate(self.decls or []):
            nodelist.append(("decls[%d]" % i, child))
        for i, child in enumerate(self.stats or []):
            nodelist.append(("stats[%d]" % i, child))
        return tuple(nodelist)

    attr_names = ()


class Return(Node):
    __slots__ = ("exp", "coord", "ref")

    def __init__(self, exp, coord=None):
        self.exp = exp
        self.coord = coord

        self.ref = None

    def children(self):
        nodelist = []
        if self.exp is not None:
            nodelist.append(("exp", self.exp))
        return tuple(nodelist)

    attr_names = ()


class Break(Node):
    __slots__ = ("coord", "ref")

    def __init__(self, coord=None):
        self.coord = coord

        self.ref = None

    def children(self):
        pass

    attr_names = ()


class Print(Node):
    __slots__ = ("exp", "coord")

    def __init__(self, exp, coord=None):
        self.exp = exp
        self.coord = coord

    def children(self):
        nodelist = []
        if self.exp is not None:
            nodelist.append(("exp", self.exp))
        return tuple(nodelist)

    attr_names = ()


class Assert(Node):
    __slots__ = ("exp", "coord")

    def __init__(self, exp, coord=None):
        self.exp = exp
        self.coord = coord

    def children(self):
        nodelist = []
        if self.exp is not None:
            nodelist.append(("exp", self.exp))
        return tuple(nodelist)

    attr_names = ()


class Read(Node):
    __slots__ = ("exps", "coord")

    def __init__(self, exps, coord=None):
        self.exps = exps
        self.coord = coord

    def children(self):
        nodelist = []
        if self.exps is not None:
            nodelist.append(("exps", self.exps))
        return tuple(nodelist)

    attr_names = ()

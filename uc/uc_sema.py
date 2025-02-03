import argparse
import pathlib
import sys
import copy

from uc.uc_ast import (
    ArrayDecl,
    ArrayRef,
    BinaryOp,
    Constant,
    ExprList,
    FuncCall,
    InitList,
    Type,
    VarDecl,
    ID,
)
from uc.uc_parser import Coord, UCParser
from uc.uc_type import (
    FuncType,
    ArrayType,
    StringType,
    ListType,
    BoolType,
    CharType,
    IntType,
    VoidType,
    uCType,
)


class SymbolTable(dict):
    """Class representing a symbol table. It should provide functionality
    for adding and looking up nodes associated with identifiers.
    """

    def __init__(self):
        super().__init__()

    def add(self, name, value):
        self[name] = value

    def lookup(self, name, scope):
        fullname = name + "." + str(scope)
        result = self.get(fullname, None)

        count = scope
        while result is None and count > 0:
            count -= 1
            fullname = name + "." + str(count)
            result = self.get(fullname, None)

        return result


class NodeVisitor:
    """A base NodeVisitor class for visiting uc_ast nodes.
    Subclass it and define your own visit_XXX methods, where
    XXX is the class name you want to visit with these
    methods.
    """

    _method_cache = None

    def visit(self, node):
        """Visit a node."""

        if self._method_cache is None:
            self._method_cache = {}

        visitor = self._method_cache.get(node.__class__.__name__, None)
        if visitor is None:
            method = "visit_" + node.__class__.__name__
            visitor = getattr(self, method, self.generic_visit)
            self._method_cache[node.__class__.__name__] = visitor

        return visitor(node)

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a
        node. Implements preorder visiting of the node.
        """
        for _, child in node.children():
            self.visit(child)


class Visitor(NodeVisitor):
    """
    Program visitor class. This class uses the visitor pattern. You need to define methods
    of the form visit_NodeName() for each kind of AST node that you want to process.
    """

    def __init__(self):
        # Initialize the symbol table
        self.symtab = SymbolTable()
        self.typemap = {
            "int": IntType,
            "char": CharType,
            "bool": BoolType,
            "string": StringType,
            "void": VoidType,
        }
        self.current_scope = 0
        self.stack_scope = []
        self.stack_loop = []
        self.stack_func = []
        self.returns = []
        self.arrayInit = []

    def _assert_semantic(self, condition, msg_code, coord, name="", ltype="", rtype=""):
        """Check condition, if false print selected error message and exit"""
        error_msgs = {
            1: f"{name} is not defined",
            2: f"subscript must be of type(int), not {ltype}",
            3: "Expression must be of type(bool)",
            4: f"Cannot assign {rtype} to {ltype}",
            5: f"Assignment operator {name} is not supported by {ltype}",
            6: f"Binary operator {name} does not have matching LHS/RHS types",
            7: f"Binary operator {name} is not supported by {ltype}",
            8: "Break statement must be inside a loop",
            9: "Array dimension mismatch",
            10: f"Size mismatch on {name} initialization",
            11: f"{name} initialization type mismatch",
            12: f"{name} initialization must be a single element",
            13: "Lists have different sizes",
            14: "List & variable have different sizes",
            15: f"conditional expression is {ltype}, not type(bool)",
            16: f"{name} is not a function",
            17: f"no. arguments to call {name} function mismatch",
            18: f"Type mismatch with parameter {name}",
            19: "The condition expression must be of type(bool)",
            20: "Expression must be a constant",
            21: "Expression is not of basic type",
            22: f"{name} does not reference a variable of basic type",
            23: f"{name} is not a variable",
            24: f"Return of {ltype} is incompatible with {rtype} function definition",
            25: f"Name {name} is already defined in this scope",
            26: f"Unary operator {name} is not supported",
            27: "Undefined error",
        }
        if not condition:
            msg = error_msgs.get(msg_code)
            print("SemanticError: %s %s" % (msg, coord), file=sys.stdout)
            sys.exit(1)

    def new_scope(self, node):
        self.current_scope += 1
        node.scope = self.current_scope
        self.stack_scope.append(node)

    def end_scope(self):
        self.stack_scope.pop()

    def new_loop_scope(self, node):
        self.new_scope(node)
        self.stack_loop.append(node)

    def end_loop_scope(self):
        self.end_scope()
        self.stack_loop.pop()

    def new_func_scope(self, node):
        self.new_scope(node)
        self.stack_func.append(node)

    def end_func_scope(self):
        self.end_scope()
        self.stack_func.pop()

    def addSafelyToSymTab(self, name, value, coord=None):
        sym = self.symtab.lookup(name, self.stack_scope[-1].scope)
        if sym is not None:
            self._assert_semantic(
                self.stack_scope[-1].scope != sym.scope, 25, coord=coord, name=name
            )
        self.symtab.add(name + "." + str(self.stack_scope[-1].scope), value)

    def getName(self, node):
        if isinstance(node, BinaryOp):
            value = node.lvalue
            while isinstance(value, BinaryOp):
                value = value.lvalue
            if isinstance(value.name, str):
                return value.name
            else:
                return self.getName(value.name)
        elif isinstance(node, ID):
            return node.name
        elif isinstance(node, Constant):
            return node
        elif isinstance(node, FuncCall):
            return node.name.name
        elif isinstance(node, ArrayRef):
            name = node.name
            while isinstance(name, ArrayRef):
                name = name.name
            return name.name
        else:
            sys.exit(1)  # Case not treated

    def findName(self, node):
        if hasattr(node, "name"):
            return node.name
        curNode = node
        while hasattr(curNode, "declname") and curNode.declname is not None:
            curNode = curNode.declname
        if not hasattr(curNode, "name") or curNode.name is None:
            sys.exit(1)  # case not treated

        return curNode

    def findArrayRefId(self, node):
        curNode = node
        while not isinstance(curNode, ID):
            curNode = curNode.name
        return curNode

    def visit_Program(self, node):
        # Visit all of the global declarations
        for _decl in node.gdecls:
            self.visit(_decl)

    def visit_GlobalDecl(self, node):
        self.new_scope(node)
        self.generic_visit(node)
        # self.end_scope() this scope never ends

    def visit_Decl(self, node):
        sameName = self.symtab.lookup(node.declname.name, self.stack_scope[-1].scope)
        if sameName is not None:
            self._assert_semantic(
                sameName.scope != self.stack_scope[-1].scope,
                25,
                coord=node.declname.coord,
                name=node.declname.name,
            )
        if isinstance(node.type, VarDecl) and node.type.init is None:
            node.type.init = node.init

        if node.init is not None:
            self.visit(node.init)
            if isinstance(node.type, ArrayDecl):
                self.arrayInit.append([node.init])

        self.visit(node.type)

        node.scope = self.stack_scope[-1].scope
        node.uc_type = node.type.uc_type

        if isinstance(node.init, InitList) and not isinstance(node.uc_type, ArrayType):
            self._assert_semantic(
                False,
                12,
                name=node.declname.name,
                coord=node.declname.coord,
            )
        if isinstance(node.uc_type, ArrayType) and node.uc_type.size is None:
            self._assert_semantic(node.init is not None, 9, coord=node.declname.coord)
            node.uc_type.size = node.init.uc_type.size
            self._assert_semantic(
                node.init.uc_type.typename == node.uc_type.typename,
                11,
                name=node.declname.name,
                coord=node.declname.coord,
            )

    def visit_VarDecl(self, node):
        self.visit(node.type)
        node.scope = self.stack_scope[-1].scope
        node.uc_type = node.type.uc_type

        if node.init is not None:
            self.visit(node.init)
            self._assert_semantic(
                node.uc_type.typename == node.init.uc_type.typename,
                11,
                name=node.declname.name,
                coord=node.declname.coord,
            )

        self.addSafelyToSymTab(node.declname.name, node.declname, node.declname.coord)
        node.declname.uc_type = node.uc_type
        node.declname.scope = self.stack_scope[-1].scope

        if len(self.arrayInit) > 0:
            for init in self.arrayInit[-1]:
                self._assert_semantic(isinstance(init, Constant), 20, coord=init.coord)
            self.arrayInit.pop()

    def visit_ArrayDecl(self, node):
        if len(self.arrayInit) > 0:
            nextInits = []
            for init in self.arrayInit[-1]:
                if isinstance(init, InitList):
                    for innerInit in init.inits:
                        nextInits.append(innerInit)
            self.arrayInit.append(nextInits)
        self.visit(node.type)

        node.scope = self.stack_scope[-1].scope
        node.uc_type = ArrayType(node.type.uc_type.typename)
        node.uc_type.inside_type = node.type.uc_type

        if node.init is not None:
            self.visit(node.init)
            self._assert_semantic(
                node.init.uc_type.typename == "int",
                2,
                ltype=node.init.uc_type.typename,
                coord=node.init.coord,
            )
            node.init.value = int(node.init.value)
            node.uc_type.size = node.init.value

            if len(self.arrayInit) > 0:
                id = self.findName(node)
                size = self.arrayInit[-1][0].uc_type.size
                for init in self.arrayInit[-1]:
                    self._assert_semantic(
                        init.uc_type.size == size,
                        13,
                        coord=id.coord,
                    )
                    size = init.uc_type.size
                for init in self.arrayInit[-1]:
                    if isinstance(init.uc_type, StringType):
                        self._assert_semantic(
                            init.uc_type.size == node.uc_type.size,
                            10,
                            name=id.name,
                            coord=id.coord,
                        )
                    else:
                        self._assert_semantic(
                            init.uc_type.size == node.uc_type.size,
                            14,
                            coord=id.coord,
                        )
        elif len(self.arrayInit) > 0:
            myInits = self.arrayInit[-1]
            if len(myInits) > 0 and isinstance(myInits[0], InitList):
                node.uc_type.size = len(myInits[0].inits)

        arrayRef = self.symtab.lookup(
            self.findName(node).name, self.stack_scope[-1].scope
        )
        arrayRef.uc_type = node.uc_type

        if len(self.arrayInit) > 0:
            self.arrayInit.pop()

    def visit_FuncDef(self, node):
        self.new_func_scope(node)
        self.visit(node.type)
        if isinstance(node.type.uc_type, FuncType):
            node.uc_type = node.type.uc_type
        else:
            node.uc_type = FuncType(node.type.uc_type)
        self.visit(node.decl)
        if node.declList is not None:
            self.visit(node.declList)

        node.compound.doNotCreateScope = True
        self.visit(node.compound)

        if len(self.returns) == 0 and node.uc_type.typename != "void":
            self._assert_semantic(
                False,
                24,
                ltype="type(void)",
                rtype="type({})".format(node.uc_type.typename),
                coord=node.compound.coord,
            )
        self.end_func_scope()

    def visit_FuncDecl(self, node):
        self.visit(node.type)

        funcOrGlobal = self.stack_scope[-1]
        node.scope = funcOrGlobal.scope
        if hasattr(funcOrGlobal, "uc_type"):
            node.uc_type = funcOrGlobal.uc_type
        else:
            node.uc_type = FuncType(node.declname.uc_type)

        if node.init is not None:
            self.visit(node.init)
            node.uc_type.params = [i.declname for i in node.init.params]

        funcRef = None
        if isinstance(node.declname, ID):
            funcRef = self.symtab.lookup(node.declname.name, self.stack_scope[-1].scope)
        else:
            funcRef = self.symtab.lookup(
                node.declname.declname.name, self.stack_scope[-1].scope
            )
        funcRef.uc_type = node.uc_type

    def visit_ParamList(self, node):
        if len(self.stack_func) > 0:
            node.ref = self.stack_func[-1]
        for param in node.params:
            self.visit(param)

    def visit_InitList(self, node):
        initsType = None
        if len(node.inits) > 0:
            self.visit(node.inits[0])
            initsType = node.inits[0].uc_type
            for init in node.inits:
                self.visit(init)
                self._assert_semantic(
                    init.uc_type.typename == initsType.typename, 27, coord=init.coord
                )
                initsType = init.uc_type

        node.uc_type = ListType(initsType.typename, len(node.inits))

    def visit_BinaryOp(self, node):
        # Visit the left and right expression
        self.visit(node.lvalue)

        ltype = node.lvalue.uc_type
        if isinstance(node.lvalue.uc_type, FuncType):
            ltype = node.lvalue.uc_type.return_type
        elif isinstance(node.lvalue.uc_type, ArrayType):
            ltype = self.typemap[node.lvalue.uc_type.typename]

        self.visit(node.rvalue)
        rtype = node.rvalue.uc_type
        if isinstance(node.rvalue.uc_type, FuncType):
            ltype = node.rvalue.uc_type.return_type
        elif isinstance(node.lvalue.uc_type, ArrayType):
            ltype = self.typemap[node.rvalue.uc_type.typename]

        # Make sure left and right operands have the same type
        self._assert_semantic(
            rtype.typename == ltype.typename, 6, name=node.op, coord=node.coord
        )
        # - Make sure the operation is supported
        self._assert_semantic(
            node.op in ltype.binary_ops
            or node.op in ltype.rel_ops
            or node.op in ltype.unary_ops,
            7,
            coord=node.coord,
            name=node.op,
            ltype=ltype.toString(),
        )
        # - Assign the result type
        if node.op in self.typemap["bool"].rel_ops:
            node.uc_type = self.typemap["bool"]
        else:
            node.uc_type = rtype

    def visit_UnaryOp(self, node):
        self.visit(node.unexp)

        self._assert_semantic(
            node.op in node.unexp.uc_type.unary_ops,
            26,
            name=node.op,
            coord=node.coord,
        )

        node.uc_type = node.unexp.uc_type

    def visit_Assignment(self, node):
        # visit right side
        self.visit(node.rvalue)
        rtype = node.rvalue.uc_type
        if isinstance(rtype, FuncType):
            rtype = rtype.return_type
        # visit left side (must be a location)
        _var = node.lvalue
        self.visit(_var)

        if isinstance(_var, ID):
            self._assert_semantic(_var.scope is not None, 1, node.coord, name=_var.name)
        ltype = _var.uc_type
        # Check that assignment is allowed
        self._assert_semantic(
            ltype.typename == rtype.typename,
            4,
            node.coord,
            ltype=ltype.toString(),
            rtype=rtype.toString(),
        )
        # Check that assign_ops is supported by the type
        self._assert_semantic(
            node.op in ltype.assign_ops,
            5,
            node.coord,
            name=node.op,
            ltype=ltype.typename,
        )

    def visit_ArrayRef(self, node):
        self.visit(node.name)
        self.visit(node.position)
        self._assert_semantic(
            node.position.uc_type.typename == "int",
            2,
            coord=node.position.coord,
            ltype=node.position.uc_type.toString(),
        )
        node.uc_type = node.name.uc_type
        node.ref = self.findArrayRefId(node)

    def visit_FuncCall(self, node):
        self.visit(node.name)
        if node.args is not None:
            self.visit(node.args)
        node.uc_type = copy.deepcopy(node.name.uc_type)

        func = self.symtab.lookup(node.name.name, self.stack_scope[-1].scope)
        self._assert_semantic(
            hasattr(func.uc_type, "params"), 16, name=func.name, coord=node.coord
        )
        definedParams = func.uc_type.params
        passedParams = []

        if node.args is not None and isinstance(node.args, ExprList):
            passedParams = node.args.exprs
        elif node.args is not None:
            passedParams = [node.args]

        self._assert_semantic(
            len(passedParams) == len(definedParams),
            17,
            name=node.name.name,
            coord=node.coord,
        )
        node.uc_type.passedParams = passedParams
        for i in range(len(definedParams)):
            name = self.getName(passedParams[i])

            self._assert_semantic(
                definedParams[i].uc_type.typename == passedParams[i].uc_type.typename,
                18,
                name=name,
                coord=passedParams[i].coord,
            )

    def visit_Constant(self, node):
        if node.type == "string":
            node.uc_type = StringType(len(node.value))
        else:
            node.uc_type = self.typemap[node.type]

        if node.type == "int":
            node.value = int(node.value)

    def visit_ID(self, node):
        var = self.symtab.lookup(node.name, self.stack_scope[-1].scope)

        self._assert_semantic(var is not None, 1, node.coord, name=node.name)
        if isinstance(var.uc_type, FuncType):
            self._assert_semantic(
                var.scope < self.current_scope,
                1,
                node.coord,
                name=node.name,
            )
        else:
            self._assert_semantic(
                var.scope in [nodeScope.scope for nodeScope in self.stack_scope],
                1,
                node.coord,
                name=node.name,
            )
        node.uc_type = var.uc_type
        node.scope = var.scope

    def visit_Type(self, node):
        if node.name == "void":
            node.uc_type = FuncType(self.typemap["void"], [])
        else:
            node.uc_type = self.typemap[node.name]

    def visit_If(self, node):
        self.new_scope(node)
        self.visit(node.exp)
        self.visit(node.istats)
        self._assert_semantic(
            hasattr(node.exp, "uc_type") and node.exp.uc_type.typename == "bool",
            19,
            node.exp.coord,
        )
        if node.estats is not None:
            self.visit(node.estats)
        self.end_scope()

    def visit_While(self, node):
        self.new_loop_scope(node)
        self.visit(node.exp)
        self._assert_semantic(
            node.exp.uc_type.typename == "bool",
            15,
            ltype="type({})".format(node.exp.uc_type.typename),
            coord=node.coord,
        )
        self.visit(node.stats)
        self.end_loop_scope()

    def visit_For(self, node):
        self.new_loop_scope(node)
        if node.left is not None:
            self.visit(node.left)
        if node.mid is not None:
            self.visit(node.mid)
            self._assert_semantic(
                hasattr(node.mid, "uc_type") and node.mid.uc_type.typename == "bool",
                19,
                node.coord,
            )
        if node.right is not None:
            self.visit(node.right)
        self.visit(node.stat)
        self.end_loop_scope()

    def visit_Compound(self, node):
        if node.doNotCreateScope != True:
            self.new_scope(node)

        self.generic_visit(node)

        if node.doNotCreateScope != True:
            self.end_scope()

    def visit_Return(self, node):
        if node.exp is not None:
            self.visit(node.exp)
        else:
            node.exp = Type("void", node.coord)
            node.exp.uc_type = self.typemap["void"]
        node.ref = self.stack_func[-1]

        self.returns.append(node)

        self._assert_semantic(
            node.exp.uc_type.typename == node.ref.uc_type.typename
            or node.ref.uc_type.typename == "void",
            24,
            ltype="type({})".format(node.exp.uc_type.typename),
            rtype="type({})".format(node.ref.uc_type.typename),
            coord=node.coord,
        )

    def visit_Break(self, node):
        self._assert_semantic(len(self.stack_loop) > 0, 8, coord=node.coord)
        node.ref = self.stack_loop[-1]

    def visit_Print(self, node):
        if node.exp is not None:
            self.visit(node.exp)
            if isinstance(node.exp, ExprList):
                for exp in node.exp.exprs:
                    name = self.getName(exp)
                    error = 21
                    symType = None
                    if isinstance(node.exp, ID) or isinstance(node.exp, ArrayRef):
                        error = 22
                        sym = self.symtab.lookup(name, self.stack_scope[-1].scope)
                        if sym is not None:
                            symType = sym.uc_type
                    self._assert_semantic(
                        (
                            exp.uc_type.typename == "int"
                            or exp.uc_type.typename == "char"
                            or exp.uc_type.typename == "string"
                        )
                        and (
                            not isinstance(symType, ArrayType)
                            or isinstance(node.exp, ArrayRef)
                        ),
                        error,
                        name=name,
                        coord=exp.coord,
                    )
            else:
                name = self.getName(node.exp)
                error = 21
                symType = None
                if isinstance(node.exp, ID) or isinstance(node.exp, ArrayRef):
                    error = 22
                    sym = self.symtab.lookup(name, self.stack_scope[-1].scope)
                    if sym is not None:
                        symType = sym.uc_type
                self._assert_semantic(
                    (
                        node.exp.uc_type.typename == "int"
                        or node.exp.uc_type.typename == "char"
                        or node.exp.uc_type.typename == "string"
                    )
                    and (
                        not isinstance(symType, ArrayType)
                        or isinstance(node.exp, ArrayRef)
                    ),
                    error,
                    name=name,
                    coord=node.exp.coord,
                )

    def visit_Assert(self, node):
        self.visit(node.exp)
        name = self.getName(node.exp)
        self._assert_semantic(
            node.exp.uc_type.typename == "bool",
            3,
            name=name,
            coord=node.exp.coord,
        )

    def visit_Read(self, node):
        self.visit(node.exps)
        if isinstance(node.exps, ExprList):
            for exp in node.exps.exprs:
                self._assert_semantic(
                    isinstance(exp, ID) or isinstance(exp, ArrayRef),
                    23,
                    name=exp,
                    coord=exp.coord,
                )
        else:
            self._assert_semantic(
                isinstance(node.exps, ID) or isinstance(node.exps, ArrayRef),
                23,
                name=node.exps,
                coord=node.exps.coord,
            )


if __name__ == "__main__":

    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file", help="Path to file to be semantically checked", type=str
    )
    args = parser.parse_args()

    # get input path
    input_file = args.input_file
    input_path = pathlib.Path(input_file)

    # check if file exists
    if not input_path.exists():
        print("Input", input_path, "not found", file=sys.stderr)
        sys.exit(1)

    # set error function
    p = UCParser()
    # open file and parse it
    with open(input_path) as f:
        ast = p.parse(f.read())
        sema = Visitor()
        sema.visit(ast)

class uCType:
    """
    Class that represents a type in the uC language.  Basic
    Types are declared as singleton instances of this type.
    """

    def __init__(
        self, name, binary_ops=set(), unary_ops=set(), rel_ops=set(), assign_ops=set()
    ):
        """
        You must implement yourself and figure out what to store.
        """
        self.typename = name
        self.unary_ops = unary_ops
        self.binary_ops = binary_ops
        self.rel_ops = rel_ops
        self.assign_ops = assign_ops

    def toString(self):
        return f"type({self.typename})"


# Create specific instances of basic types. You will need to add
# appropriate arguments depending on your definition of uCType
IntType = uCType(
    "int",
    unary_ops={"-", "+"},
    binary_ops={"+", "-", "*", "/", "%"},
    rel_ops={"==", "!=", "<", ">", "<=", ">="},
    assign_ops={"="},
)

CharType = uCType(
    "char",
    unary_ops={},
    binary_ops={},
    rel_ops={"==", "!=", "<", ">", "<=", ">="},
    assign_ops={"="},
)

BoolType = uCType(
    "bool",
    unary_ops={"!"},
    binary_ops={"&&", "||"},
    rel_ops={"==", "!=", "<", ">", "<=", ">="},
    assign_ops={},
)

VoidType = uCType(
    "void",
    unary_ops={},
    binary_ops={},
    rel_ops={},
    assign_ops={},
)


class StringType(uCType):
    def __init__(self, size=None):
        """
        type: Any of the uCTypes can be used as the array's type. This
              means that there's support for nested types, like matrices.
        size: Integer with the length of the array.
        """
        self.size = size
        super().__init__(
            "string", assign_ops={"="}, unary_ops={"*", "&"}, rel_ops={"==", "!="}
        )


class ArrayType(uCType):
    def __init__(self, element_type, size=None, inside_type=None):
        """
        type: Any of the uCTypes can be used as the array's type. This
              means that there's support for nested types, like matrices.
        size: Integer with the length of the array.
        """
        self.size = size
        self.inside_type = inside_type
        super().__init__(
            element_type, assign_ops={"="}, unary_ops={"*", "&"}, rel_ops={"==", "!="}
        )


class FuncType(uCType):
    def __init__(self, return_type, params=None):
        """
        type: Any of the uCTypes can be used as the array's type. This
              means that there's support for nested types, like matrices.
        size: Integer with the length of the array.
        """
        self.return_type = return_type
        self.params = params
        self.passedParams = None
        super().__init__(return_type.typename)


class ListType(uCType):
    def __init__(self, basicType, size=None):
        """
        type: Any of the uCTypes can be used as the array's type. This
              means that there's support for nested types, like matrices.
        size: Integer with the length of the array.
        """
        self.size = size
        super().__init__(
            basicType, assign_ops={"="}, unary_ops={"*", "&"}, rel_ops={"==", "!="}
        )

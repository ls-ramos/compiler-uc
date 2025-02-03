import argparse
import pathlib
import sys
import re
from uc.uc_ast import ArrayDecl, ArrayRef, Constant, ExprList, For, FuncDef, ID, While
from uc.uc_block import CFG, BasicBlock, ConditionBlock, EmitBlocks, format_instruction
from uc.uc_interpreter import Interpreter
from uc.uc_parser import UCParser
from uc.uc_sema import NodeVisitor, Visitor
from uc.uc_type import FuncType, ArrayType


class CodeGenerator(NodeVisitor):
    """
    Node visitor class that creates 3-address encoded instruction sequences
    with Basic Blocks & Control Flow Graph.
    """

    def __init__(self, viewcfg):
        self.viewcfg = viewcfg
        self.current_block = None

        # version dictionary for temporaries. We use the name as a Key
        self.fname = "_glob_"
        self.versions = {self.fname: 1}

        # The generated code (list of tuples)
        # At the end of visit_program, we call each function definition to emit
        # the instructions inside basic blocks. The global instructions that
        # are stored in self.text are appended at beginning of the code
        self.code = []

        self.text = []  # Used for global declarations & constants (list, strings)

        self.binary_ops = {
            "+": "add",
            "-": "sub",
            "*": "mul",
            "/": "div",
            "%": "mod",
            ">=": "ge",
            "<=": "le",
            "==": "eq",
            ">": "gt",
            "<": "lt",
            "!=": "ne",
            "&&": "and",
            "||": "or",
            "!": "not",
        }

        self.arrayInit = None
        self.returnTarget = None

        self.globalVars = []
        self.falseAssertBlocks = []

        self.forBlocksCount = 0
        self.whileBlocksCount = 0
        self.ifBlocksCount = 0
        self.assertBlocksCount = 0
        self.stringConstCount = 0

        self.arrayRefAssign = False
        self.endBlockLoopStack = []

    def show(self, buf=sys.stdout):
        _str = ""
        for _code in self.code:
            _str += format_instruction(_code) + "\n"
        buf.write(_str)

    def new_temp(self):
        """
        Create a new temporary variable of a given scope (function name).
        """
        if self.fname not in self.versions:
            self.versions[self.fname] = 1
        name = "%" + "%d" % (self.versions[self.fname])
        self.versions[self.fname] += 1
        return name

    def new_text(self, typename):
        """
        Create a new literal constant on global section (text).
        """
        name = "@." + typename + "." + "%d" % (self.versions["_glob_"])
        self.versions["_glob_"] += 1
        return name

    def getId(self, node):
        while not isinstance(node, ID):
            node = node.declname
        return node

    def fixArrayInit(self, list):
        result = []
        for item in list:
            if isinstance(item, Constant):
                result.append(item.value)
            else:
                result.append(self.fixArrayInit(item.inits))
        return result

    # You must implement visit_Nodename methods for all of the other
    # AST nodes.  In your code, you will need to make instructions
    # and append them to the current block code list.
    #
    # A few sample methods follow. Do not hesitate to complete or change
    # them if needed.

    def visit_Program(self, node):
        # Visit all of the global declarations
        for _decl in node.gdecls:
            self.visit(_decl)
        # At the end of codegen, first init the self.code with
        # the list of global instructions allocated in self.text
        self.code = self.text.copy()
        # Also, copy the global instructions into the Program node
        node.text = self.text.copy()
        # After, visit all the function definitions and emit the
        # code stored inside basic blocks.
        for _decl in node.gdecls:
            if isinstance(_decl, FuncDef):
                # _decl.cfg contains the Control Flow Graph for the function
                # cfg points to start basic block
                bb = EmitBlocks()
                bb.visit(_decl.cfg)
                for _code in bb.code:
                    self.code.append(_code)

        if self.viewcfg:  # evaluate to True if -cfg flag is present in command line
            for _decl in node.gdecls:
                if isinstance(_decl, FuncDef):
                    dot = CFG(_decl.decl.declname.name)
                    dot.view(_decl.cfg)  # _decl.cfg contains the CFG for the function

    def visit_FuncDef(self, node):
        node.cfg = BasicBlock("entry")
        self.current_block = node.cfg

        self.visit(node.decl)
        self.visit(node.compound)

        exit = BasicBlock("exit")
        exit.append(("exit:",))
        self.current_block.append(("jump", "%" + "exit"))
        self.current_block.next_block = exit
        self.current_block.branch = exit
        self.current_block = self.current_block.next_block

        if node.uc_type.return_type.typename == "void":
            self.current_block.append(("return_void",))
        else:
            target = self.new_temp()
            self.current_block.append(
                ("load_" + node.uc_type.return_type.typename, self.returnTarget, target)
            )
            self.current_block.append(
                ("return_" + node.uc_type.return_type.typename, target)
            )

        # fix branch of assert fails
        for fail in self.falseAssertBlocks:
            if fail.instructions[-1][0] != "jump":
                fail.append(("jump", "%" + "exit"))
            fail.branch = exit

    def visit_Decl(self, node):
        if isinstance(node.type, ArrayDecl):
            self.arrayInit = node.init
        self.visit(node.type)

    def visit_FuncDecl(self, node):
        if self.current_block is not None:
            # alloc temp for each params
            temps = []
            if node.init is not None:
                temps = [
                    (param.declname.uc_type.typename, self.new_temp())
                    for param in node.init.params
                ]

            inst = (
                "define_" + node.uc_type.typename,
                "@" + self.getId(node).name,
                temps,
            )
            self.current_block.append(inst)
            self.current_block.append(("entry:",))
            if node.init is not None:
                self.visit(node.init)

                # store the temps in the param variables
                for temp, param in zip(temps, node.init.params):
                    self.current_block.append(
                        ("store_" + temp[0], temp[1], param.declname.gen_location)
                    )

            if node.uc_type.return_type.typename != "void":
                targetRet = self.new_temp()
                self.current_block.append(
                    ("alloc_" + node.uc_type.return_type.typename, targetRet)
                )
                self.returnTarget = targetRet

    def visit_ArrayDecl(self, node):
        self.visit(node.type)

    def visit_VarDecl(self, node):
        # Allocate on stack memory
        if self.current_block is None and isinstance(node.declname.uc_type, ArrayType):
            inst = (
                "global_"
                + node.declname.uc_type.typename
                + "_"
                + str(node.declname.uc_type.size),
                "@" + node.declname.name,
                self.fixArrayInit(self.arrayInit.inits),
            )
            self.text.append(inst)
            self.globalVars.append(node.declname.name)
        elif self.current_block is None:
            inst = (
                "global_" + node.declname.uc_type.typename,
                "@" + node.declname.name,
                node.init.value,
            )
            self.text.append(inst)
            self.globalVars.append(node.declname.name)
        elif (
            isinstance(node.declname.uc_type, ArrayType) and self.arrayInit is not None
        ):
            gname = "@.const." + node.declname.name + "." + str(node.declname.scope)
            lname = "%" + node.declname.name + "." + str(node.declname.scope)
            arrType = (
                node.declname.uc_type.typename + "_" + str(node.declname.uc_type.size)
            )
            inside = node.declname.uc_type.inside_type
            while isinstance(inside, ArrayType):
                arrType += "_" + str(inside.size)
                inside = inside.inside_type

            if isinstance(self.arrayInit, Constant):
                inst = (
                    "global_string",
                    gname,
                    self.arrayInit.value,
                )
            else:
                inst = (
                    "global_" + arrType,
                    gname,
                    self.fixArrayInit(self.arrayInit.inits),
                )
            self.text = [inst] + self.text
            self.current_block.append(("alloc_" + arrType, lname))
            source = gname
            if "@" in gname:
                source = self.new_temp()
                self.current_block.append(("load_" + arrType, gname, source))
            self.current_block.append(("store_" + arrType, source, lname))
        elif isinstance(node.declname.uc_type, ArrayType):
            self.visit(node.declname)
            arrType = (
                node.declname.uc_type.typename + "_" + str(node.declname.uc_type.size)
            )
            inside = node.declname.uc_type.inside_type
            while isinstance(inside, ArrayType):
                arrType += "_" + str(inside.size)
                inside = inside.inside_type
            inst = (
                "alloc_" + arrType,
                node.declname.gen_location,
            )
            self.current_block.append(inst)
        else:
            self.visit(node.declname)

            inst = ("alloc_" + node.type.name, node.declname.gen_location)
            self.current_block.append(inst)

            # Store optional init val
            _init = node.init
            if _init is not None:
                self.visit(_init)
                source = _init.gen_location
                if "@" in _init.gen_location or isinstance(_init, ID):
                    source = self.new_temp()
                    self.current_block.append(
                        (
                            "load_" + node.type.name,
                            _init.gen_location,
                            source,
                        )
                    )

                inst = (
                    "store_" + node.type.name,
                    source,
                    node.declname.gen_location,
                )
                self.current_block.append(inst)

    def visit_FuncCall(self, node):
        # load params
        temps = []
        if node.uc_type.passedParams is not None:
            for param in node.uc_type.passedParams:
                self.visit(param)

                target = self.new_temp()
                paramType = param.uc_type.typename

                self.current_block.append(
                    ("load_" + paramType, param.gen_location, target)
                )
                temps.append((paramType, target))

        # pass params
        for param in temps:
            self.current_block.append(("param_" + param[0], param[1]))

        # call function
        if node.uc_type.return_type.typename != "void":
            returnTarget = self.new_temp()
            self.current_block.append(
                (
                    "call_" + node.uc_type.return_type.typename,
                    "@" + node.name.name,
                    returnTarget,
                )
            )
            node.gen_location = returnTarget
        else:
            self.current_block.append(
                ("call_" + node.uc_type.return_type.typename, "@" + node.name.name)
            )

    def calculatePos(self, node, inside_type, insts, count):
        nameGen = None
        prevTarget = None
        if isinstance(node.name, ArrayRef):
            (prevTarget, nameGen) = self.calculatePos(
                node.name, inside_type.inside_type, insts, count + 1
            )
        else:
            self.visit(node.name)
            nameGen = node.name.gen_location

        self.visit(node.position)
        targetPos = None
        if isinstance(node.position, ID):
            targetPos = self.new_temp()
            insts.append(
                (
                    "load_" + node.position.uc_type.typename,
                    node.position.gen_location,
                    targetPos,
                )
            )
        else:
            targetPos = node.position.gen_location

        if count != 0:  # not first interaction
            targetSize = self.new_temp()
            insts.append(("literal_int", inside_type.size, targetSize))
            targetMul = self.new_temp()
            insts.append(("mul_int", targetSize, targetPos, targetMul))
            targetPos = targetMul

        if prevTarget is not None:
            targetFinal = self.new_temp()
            insts.append(("add_int", prevTarget, targetPos, targetFinal))
            targetPos = targetFinal

        return (targetPos, nameGen)

    def visit_ArrayRef(self, node):

        nameGenLocation = None
        targetPos = None

        if not isinstance(node.name, ArrayRef):
            self.visit(node.position)
            self.visit(node.name)
            nameGenLocation = node.name.gen_location

            if isinstance(node.position, ID):
                targetPos = self.new_temp()
                self.current_block.append(
                    (
                        "load_" + node.position.uc_type.typename,
                        node.position.gen_location,
                        targetPos,
                    )
                )
            else:
                targetPos = node.position.gen_location
        else:
            insts = []
            (targetPos, nameGenLocation) = self.calculatePos(
                node, node.uc_type, insts, 0
            )
            for inst in insts:
                self.current_block.append(inst)

        targetArr = self.new_temp()
        self.current_block.append(
            (
                "elem_" + node.uc_type.typename,
                nameGenLocation,
                targetPos,
                targetArr,
            )
        )
        if not self.arrayRefAssign:
            targetFin = self.new_temp()
            self.current_block.append(
                ("load_" + node.uc_type.typename + "_*", targetArr, targetFin)
            )
            node.gen_location = targetFin
        else:
            node.gen_location = targetArr

        self.arrayRefAssign = False

    def visit_Constant(self, node):
        if node.type == "string":
            _target = self.new_text("str")
            inst = ("global_string", _target, node.value)
            self.text.append(inst)
        else:
            # Create a new temporary variable name
            _target = self.new_temp()
            # Make the SSA opcode and append to list of generated instructions
            inst = ("literal_" + node.uc_type.typename, node.value, _target)
            self.current_block.append(inst)
        # Save the name of the temporary variable where the value was placed
        node.gen_location = _target

    def create_print(self, exp):
        if isinstance(exp, ExprList):
            for ex in exp.exprs:
                self.create_print(ex)
        elif isinstance(exp, ID):
            self.visit(exp)
            var = self.new_temp()
            inst = ("load_" + exp.uc_type.typename, exp.gen_location, var)
            self.current_block.append(inst)
            inst = ("print_" + exp.uc_type.typename, var)
            self.current_block.append(inst)
        elif exp is not None:
            self.visit(exp)
            inst = ("print_" + exp.uc_type.typename, exp.gen_location)
            self.current_block.append(inst)
        else:
            inst = ("print_void",)
            self.current_block.append(inst)

    def visit_Print(self, node):
        self.create_print(node.exp)

    def visit_ID(self, node):
        if node.name in self.globalVars:
            node.gen_location = "@" + node.name
        else:
            node.gen_location = "%" + node.name + "." + str(node.scope)

    def visit_Assignment(self, node):
        self.visit(node.rvalue)

        if isinstance(node.rvalue, ID):
            target = self.new_temp()
            self.current_block.append(
                (
                    "load_" + node.rvalue.uc_type.typename,
                    node.rvalue.gen_location,
                    target,
                )
            )
        else:
            target = node.rvalue.gen_location

        if isinstance(node.lvalue, ArrayRef):
            self.arrayRefAssign = True
            self.visit(node.lvalue)
            source = target
            if "@" in target:
                source = self.new_temp()
                self.current_block.append(
                    ("load_" + node.lvalue.uc_type.typename + "_*", target, source)
                )
            self.current_block.append(
                (
                    "store_" + node.lvalue.uc_type.typename + "_*",
                    source,
                    node.lvalue.gen_location,
                )
            )
        else:
            self.visit(node.lvalue)
            source = target
            if "@" in target:
                source = self.new_temp()
                self.current_block.append(
                    ("load_" + node.lvalue.uc_type.typename, target, source)
                )
            self.current_block.append(
                (
                    "store_" + node.lvalue.uc_type.typename,
                    source,
                    node.lvalue.gen_location,
                )
            )

    def visit_Break(self, node):
        if isinstance(node.ref, For):
            self.current_block.append(
                ("jump", "%" + "for.end." + str(self.forBlocksCount))
            )
        elif isinstance(node.ref, While):
            self.current_block.append(
                ("jump", "%" + "while.end." + str(self.whileBlocksCount))
            )
        self.endBlockLoopStack[-1].breaks.append(self.current_block)

    def visit_BinaryOp(self, node):
        # Visit the lvalue and rvalue expressions
        self.visit(node.lvalue)
        self.visit(node.rvalue)
        if isinstance(node.lvalue, ID):  # TODO: add ArrayRef too
            ltarget = self.new_temp()
            self.current_block.append(
                (
                    "load_" + node.lvalue.uc_type.typename,
                    node.lvalue.gen_location,
                    ltarget,
                )
            )
        else:
            ltarget = node.lvalue.gen_location

        if isinstance(node.rvalue, ID):  # TODO: add ArrayRef too
            rtarget = self.new_temp()
            self.current_block.append(
                (
                    "load_" + node.rvalue.uc_type.typename,
                    node.rvalue.gen_location,
                    rtarget,
                )
            )
        else:
            rtarget = node.rvalue.gen_location

        # Make a new temporary for storing the result
        target = self.new_temp()

        # Create the opcode and append to list
        opcode = self.binary_ops[node.op] + "_" + node.lvalue.uc_type.typename
        inst = (opcode, ltarget, rtarget, target)
        self.current_block.append(inst)

        # Store location of the result on the node
        node.gen_location = target

    def visit_Return(self, node):
        if node.exp is not None and node.ref.uc_type.return_type.typename != "void":
            self.visit(node.exp)
            source = node.exp.gen_location
            if "@" in node.exp.gen_location or isinstance(node.exp, ID):
                source = self.new_temp()
                self.current_block.append(
                    (
                        "load_" + node.ref.uc_type.return_type.typename,
                        node.exp.gen_location,
                        source,
                    )
                )
            self.current_block.append(
                (
                    "store_" + node.ref.uc_type.return_type.typename,
                    source,
                    self.returnTarget,
                )
            )
        # TODO: branch to the exit label

    def visit_For(self, node):
        # added for block
        self.forBlocksCount += 1

        self.visit(node.left)

        condBlock = ConditionBlock("for.cond." + str(self.forBlocksCount))
        condBlock.append((condBlock.label + ":",))
        bodyBlock = BasicBlock("for.body." + str(self.forBlocksCount))
        bodyBlock.append((bodyBlock.label + ":",))
        incBlock = BasicBlock("for.inc." + str(self.forBlocksCount))
        incBlock.append((incBlock.label + ":",))
        endBlock = BasicBlock("for.end." + str(self.forBlocksCount))
        endBlock.append((endBlock.label + ":",))

        self.endBlockLoopStack.append(endBlock)

        # cond block
        self.current_block.append(("jump", "%" + condBlock.label))
        self.current_block.next_block = condBlock
        self.current_block.branch = condBlock
        self.current_block = condBlock
        self.visit(node.mid)
        self.current_block.append(
            (
                "cbranch",
                node.mid.gen_location,
                "%" + bodyBlock.label,
                "%" + endBlock.label,
            )
        )
        self.current_block.next_block = bodyBlock
        self.current_block.taken = bodyBlock
        self.current_block.fall_through = endBlock

        # body block
        self.current_block = bodyBlock
        # self.current_block.append((bodyBlock.label + ":",))
        self.visit(node.stat)
        self.current_block.append(("jump", "%" + incBlock.label))
        self.current_block.next_block = incBlock
        self.current_block.branch = incBlock

        # inc block
        self.current_block = incBlock
        # self.current_block.append((incBlock.label + ":",))
        self.visit(node.right)
        self.current_block.append(("jump", "%" + condBlock.label))
        self.current_block.next_block = endBlock
        self.current_block.branch = condBlock

        # end block
        self.current_block = endBlock
        self.endBlockLoopStack.pop()
        for block in endBlock.breaks:
            block.branch = endBlock

    def visit_While(self, node):
        # added while block
        self.whileBlocksCount += 1

        condBlock = ConditionBlock("while.cond." + str(self.whileBlocksCount))
        condBlock.append((condBlock.label + ":",))
        bodyBlock = BasicBlock("while.body." + str(self.whileBlocksCount))
        bodyBlock.append((bodyBlock.label + ":",))
        endBlock = BasicBlock("while.end." + str(self.whileBlocksCount))
        endBlock.append((endBlock.label + ":",))

        self.endBlockLoopStack.append(endBlock)

        # cond block
        self.current_block.append(("jump", "%" + condBlock.label))
        self.current_block.next_block = condBlock
        self.current_block.branch = condBlock
        self.current_block = condBlock
        self.visit(node.exp)
        self.current_block.append(
            (
                "cbranch",
                node.exp.gen_location,
                "%" + bodyBlock.label,
                "%" + endBlock.label,
            )
        )
        self.current_block.next_block = bodyBlock
        self.current_block.taken = bodyBlock
        self.current_block.fall_through = endBlock

        # body block
        self.current_block = bodyBlock
        self.visit(node.stats)
        self.current_block.append(("jump", "%" + condBlock.label))
        self.current_block.next_block = endBlock
        self.current_block.branch = condBlock

        # end block
        self.current_block = endBlock

        self.endBlockLoopStack.pop()
        for block in endBlock.breaks:
            block.branch = endBlock

    def visit_If(self, node):
        self.ifBlocksCount += 1

        ifThen = BasicBlock("if.then." + str(self.ifBlocksCount))
        ifThen.append((ifThen.label + ":",))

        ifEnd = BasicBlock("if.end." + str(self.ifBlocksCount))
        ifEnd.append((ifEnd.label + ":",))

        ifCond = ConditionBlock("if.cond." + str(self.ifBlocksCount))
        ifCond.append((ifCond.label + ":",))

        self.current_block.append(("jump", "%" + ifCond.label))
        self.current_block.next_block = ifCond
        self.current_block.branch = ifCond
        self.current_block = ifCond
        self.visit(node.exp)

        self.current_block = ifThen
        self.visit(node.istats)

        if ifThen.branch is None:
            if ifThen.instructions[-1][0] != "jump":
                ifThen.append(("jump", "%" + ifEnd.label))
                ifThen.branch = ifEnd
        else:
            self.current_block.append(("jump", "%" + ifEnd.label))
            self.current_block.branch = ifEnd

        if node.estats is not None:

            ifElse = BasicBlock("if.else." + str(self.ifBlocksCount))
            ifElse.append((ifElse.label + ":",))

            ifThen.next_block = ifElse

            self.current_block = ifElse
            self.visit(node.estats)
            ifElse.append(("jump", "%" + ifEnd.label))
            ifElse.next_block = ifEnd
            ifElse.branch = ifEnd

            ifCond.append(
                (
                    "cbranch",
                    node.exp.gen_location,
                    "%" + ifThen.label,
                    "%" + ifElse.label,
                )
            )
            ifCond.next_block = ifThen
            ifCond.taken = ifThen
            ifCond.fall_through = ifElse

            self.current_block = ifEnd
        else:
            if ifThen.next_block is None:
                ifThen.next_block = ifEnd
            else:
                self.current_block.next_block = ifEnd

            ifCond.append(
                (
                    "cbranch",
                    node.exp.gen_location,
                    "%" + ifThen.label,
                    "%" + ifEnd.label,
                )
            )
            ifCond.next_block = ifThen
            ifCond.taken = ifThen
            ifCond.fall_through = ifEnd

            self.current_block = ifEnd

    def visit_Assert(self, node):
        trueAssert = BasicBlock("assert.true." + str(self.assertBlocksCount))
        trueAssert.append((trueAssert.label + ":",))

        # define the fail assert text
        failStrName = self.new_text("str")
        failTextInst = (
            "global_string",
            failStrName,
            "assertion_fail on {0}:{1}".format(
                node.exp.coord.line, node.exp.coord.column
            ),
        )
        self.text.append(failTextInst)

        falseAssert = BasicBlock("assert.false." + str(self.assertBlocksCount))
        falseAssert.append((falseAssert.label + ":",))
        falseAssert.append(("print_string", failStrName))
        falseAssert.append(("jump", "%" + "exit"))  # end of function
        falseAssert.next_block = trueAssert
        self.falseAssertBlocks.append(
            falseAssert
        )  # save false asserts to set their branch later

        condBlock = ConditionBlock("assert.cond." + str(self.assertBlocksCount))
        self.current_block.append(("jump", "%" + condBlock.label))
        self.current_block.next_block = condBlock
        self.current_block.branch = condBlock
        condBlock.append((condBlock.label + ":",))
        self.current_block = condBlock
        self.visit(node.exp)
        condBlock.append(
            (
                "cbranch",
                node.exp.gen_location,
                "%" + trueAssert.label,
                "%" + falseAssert.label,
            )
        )
        condBlock.next_block = falseAssert
        condBlock.taken = trueAssert
        condBlock.fall_through = falseAssert

        self.current_block = trueAssert

        # added assert block
        self.assertBlocksCount += 1

    def visit_UnaryOp(self, node):
        self.visit(node.unexp)
        if node.op == "!":
            target = self.new_temp()
            self.current_block.append(
                (
                    self.binary_ops[node.op] + "_" + node.uc_type.typename,
                    node.unexp.gen_location,
                    target,
                )
            )
            node.gen_location = target
        elif node.op == "-":
            zeroTarget = self.new_temp()
            self.current_block.append(
                (
                    "literal_int",
                    0,
                    zeroTarget,
                )
            )
            target = self.new_temp()
            self.current_block.append(
                (
                    self.binary_ops[node.op] + "_" + node.uc_type.typename,
                    zeroTarget,
                    node.unexp.gen_location,
                    target,
                )
            )
            node.gen_location = target


if __name__ == "__main__":

    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        help="Path to file to be used to generate uCIR. By default, this script only runs the interpreter on the uCIR. \
              Use the other options for printing the uCIR, generating the CFG or for the debug mode.",
        type=str,
    )
    parser.add_argument(
        "--ir",
        help="Print uCIR generated from input_file.",
        action="store_true",
    )
    parser.add_argument(
        "--cfg", help="Show the cfg of the input_file.", action="store_true"
    )
    parser.add_argument(
        "--debug", help="Run interpreter in debug mode.", action="store_true"
    )
    args = parser.parse_args()

    print_ir = args.ir
    create_cfg = args.cfg
    interpreter_debug = args.debug

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

    gen = CodeGenerator(create_cfg)
    gen.visit(ast)
    gencode = gen.code

    if print_ir:
        print("Generated uCIR: --------")
        gen.show()
        print("------------------------\n")

    vm = Interpreter(interpreter_debug)
    vm.run(gencode)

import argparse
import pathlib
import sys
import re
from uc.uc_ast import FuncDef
from uc.uc_block import (
    BasicBlock,
    CFG,
    ConditionBlock,
    ReachDefinitions,
    format_instruction,
    EmitBlocks,
)
from uc.uc_code import CodeGenerator
from uc.uc_interpreter import Interpreter
from uc.uc_parser import UCParser
from uc.uc_sema import NodeVisitor, Visitor

assignment = (
    "store",
    "load",
    "literal",
    "elem",
    "add",
    "sub",
    "mul",
    "div",
    "mod",
    "lt",
    "le",
    "ge",
    "gt",
    "eq",
    "ne",
    "and",
    "or",
    "not",
    "call",
    "read",
)


class DataFlow(NodeVisitor):
    def __init__(self, viewcfg):
        # flag to show the optimized control flow graph
        self.viewcfg = viewcfg
        # list of code instructions after optimizations
        self.code = []
        self.rd_blocks = []

    def show(self, buf=sys.stdout):
        _str = ""
        for _code in self.code:
            _str += format_instruction(_code) + "\n"
        buf.write(_str)

    def appendOptimizedCode(self, cfg):
        bb = EmitBlocks()
        bb.visit(cfg)
        for _code in bb.code:
            self.code.append(_code)

    def addSuccPred(self, block):
        if isinstance(block, BasicBlock):
            if block.next_block is not None:
                block.successors.append(block.next_block)
                block.next_block.predecessors.append(block)
            if block.branch is not None and block.branch.id != block.next_block.id:
                block.successors.append(block.branch)
                block.branch.predecessors.append(block)
        elif isinstance(block, ConditionBlock):
            if block.next_block is not None:
                block.successors.append(block.next_block)
                block.next_block.predecessors.append(block)
            if block.taken is not None and block.taken.id != block.next_block.id:
                block.successors.append(block.taken)
                block.taken.predecessors.append(block)
            if (
                block.fall_through is not None
                and block.fall_through.id != block.next_block.id
                and block.fall_through.id != block.taken.id
            ):
                block.successors.append(block.fall_through)
                block.fall_through.predecessors.append(block)

    def buildRD_blocks(self, cfg):
        count = 0
        self.rd_blocks = []
        while cfg is not None:
            self.rd_blocks.append(cfg)
            cfg.id = count
            count += 1
            cfg = cfg.next_block

        for block in self.rd_blocks:
            self.addSuccPred(block)

    def get_operation(self, inst):
        # extract the operation & qualifier from the instruction
        return inst[0].split("_")[0]

    def computeRD_gen_kill(self):
        # Get all the definitions
        defs = {}
        for bb in self.rd_blocks:
            for inst_count, inst in enumerate(bb.instructions):
                op_code = self.get_operation(inst)
                if op_code in assignment:
                    target = inst[-1]
                    if target not in defs.keys():
                        defs[target] = [(bb.id, inst_count, target)]
                    else:
                        defs[target].append((bb.id, inst_count, target))

        # Generate the gen and kill
        for bb in self.rd_blocks:
            for inst_count, inst in enumerate(bb.instructions):
                op_code = self.get_operation(inst)
                bb.insts_rd.append(ReachDefinitions())
                if op_code in assignment:
                    target = inst[-1]

                    kills = set(defs[target]) - set([(bb.id, inst_count, target)])
                    bb.insts_rd[inst_count].kill = kills  # save kill of instruction
                    bb.rd.kill = bb.rd.kill.union(kills)

                    gen = set([(bb.id, inst_count, target)])
                    bb.insts_rd[inst_count].gen = gen  # save gen of instruction
                    bb.rd.gen = gen.union(bb.rd.gen - kills)

    def computeRD_in_out(self):
        changed = set()

        # put all blocks into the changed set
        for block in self.rd_blocks:
            changed = changed.union(set([block.id]))

        while len(changed) > 0:
            bb_pos = changed.pop()
            if bb_pos is None:  # TODO: remove that after fix bug in t15.in
                continue
            bb = self.rd_blocks[bb_pos]

            # calculate IN[b] from predecessors' OUT[p]
            for pb in bb.predecessors:
                bb.rd.ins = bb.rd.ins.union(pb.rd.out)

            oldout = bb.rd.out

            # update OUT[b] using transfer function f_b ()
            bb.rd.out = bb.rd.gen.union(bb.rd.ins - bb.rd.kill)

            # any change to OUT[b] compared to previous value?
            if oldout != bb.rd.out:  # compare oldout vs. OUT[b]
                # if yes, put all successors of b into the changed set
                for sb in bb.successors:
                    changed = changed.union(set([sb.id]))

    def computeInst_in_out(self):
        for block in self.rd_blocks:
            for count in range(len(block.instructions)):
                inst_rd = block.insts_rd[count]
                if count == 0:
                    inst_rd.ins = block.rd.ins
                else:
                    inst_rd.ins = block.insts_rd[count - 1].out

                inst_rd.out = inst_rd.gen.union(inst_rd.ins - inst_rd.kill)

    def get_stores(self, ins_list, var):
        result = []
        for ins in ins_list:
            if (
                ins[-1] == var
                and self.get_operation(self.rd_blocks[ins[0]].instructions[ins[1]])
                == "store"
            ):
                result.append(ins)
        return result

    def propagate_store_value(self):
        change = {}
        for block in self.rd_blocks:
            for i, inst in enumerate(block.instructions):
                target = inst[-1]
                if len(change) > 0:
                    new_tuple = []
                    for j in range(len(inst)):
                        if inst[j] in change.keys():
                            new_tuple.append(change[inst[j]])
                        else:
                            new_tuple.append(inst[j])
                    block.instructions[i] = tuple(new_tuple)

                if self.get_operation(inst) == "load":
                    var = inst[1]
                    var_stores = self.get_stores(block.insts_rd[i].ins, var)

                    if len(var_stores) == 1:
                        var_store = var_stores[0]
                        value_stored = self.rd_blocks[var_store[0]].instructions[
                            var_store[1]
                        ][1]
                        if "@" not in value_stored:
                            change[target] = value_stored

    def constant_propagation(self):
        self.propagate_store_value()

    def removeUnsedAssigns(self):
        vars = {}
        var_inst = {}
        for k, block in enumerate(self.rd_blocks):
            for j, inst in enumerate(block.instructions):
                op = self.get_operation(inst)
                if op == "define":
                    continue
                target = inst[-1]
                if re.match("%\d+", target) and not target in vars.keys():
                    vars[target] = 0
                    var_inst[target] = (k, inst)

                for i in range(len(inst)):
                    if inst[i] in vars.keys():
                        vars[inst[i]] += 1
        for var in vars.keys():
            if vars[var] == 1:
                remo = var_inst[var]
                self.rd_blocks[remo[0]].instructions.remove(remo[1])

    def deadcode_elimination(self):
        self.removeUnsedAssigns()

    def visit_Program(self, node):
        # First, save the global instructions on code member
        self.code = node.text[:]  # [:] to do a copy
        for _decl in node.gdecls:
            if isinstance(_decl, FuncDef):
                # start with Reach Definitions Analysis
                self.buildRD_blocks(_decl.cfg)
                # self.computeRD_gen_kill()
                # self.computeRD_in_out()
                # self.computeInst_in_out()
                # # and do constant propagation optimization
                # self.constant_propagation()

                # # # after do live variable analysis
                # # self.buildLV_blocks(_decl.cfg)
                # # self.computeLV_use_def()
                # # self.computeLV_in_out()
                # # # and do dead code elimination
                # self.deadcode_elimination()

                # # after that do cfg simplify (optional)
                # # self.short_circuit_jumps(_decl.cfg)
                # # self.merge_blocks(_decl.cfg)
                # # self.discard_unused_allocs(_decl.cfg)

                # # finally save optimized instructions in self.code
                self.appendOptimizedCode(_decl.cfg)

        if self.viewcfg:
            for _decl in node.gdecls:
                if isinstance(_decl, FuncDef):
                    dot = CFG(_decl.decl.declname.name + ".opt")
                    dot.view(_decl.cfg)


if __name__ == "__main__":

    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        help="Path to file to be used to generate uCIR. By default, this script runs the interpreter on the optimized uCIR \
              and shows the speedup obtained from comparing original uCIR with its optimized version.",
        type=str,
    )
    parser.add_argument(
        "--opt",
        help="Print optimized uCIR generated from input_file.",
        action="store_true",
    )
    parser.add_argument(
        "--speedup",
        help="Show speedup from comparing original uCIR with its optimized version.",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--debug", help="Run interpreter in debug mode.", action="store_true"
    )
    parser.add_argument(
        "-c",
        "--cfg",
        help="show the CFG of the optimized uCIR for each function in pdf format",
        action="store_true",
    )
    args = parser.parse_args()

    speedup = args.speedup
    print_opt_ir = args.opt
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

    gen = CodeGenerator(False)
    gen.visit(ast)
    gencode = gen.code

    opt = DataFlow(create_cfg)
    opt.visit(ast)
    optcode = opt.code
    if print_opt_ir:
        print("Optimized uCIR: --------")
        opt.show()
        print("------------------------\n")

    speedup = len(gencode) / len(optcode)
    sys.stderr.write(
        "[SPEEDUP] Default: %d Optimized: %d Speedup: %.2f\n\n"
        % (len(gencode), len(optcode), speedup)
    )

    vm = Interpreter(interpreter_debug)
    vm.run(optcode)

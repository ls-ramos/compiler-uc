import argparse
import pathlib
import sys
from ctypes import CFUNCTYPE, alignment, c_int
from llvmlite import binding, ir
from uc.uc_ast import FuncDef
from uc.uc_block import BlockVisitor
from uc.uc_code import CodeGenerator
from uc.uc_parser import UCParser
from uc.uc_sema import NodeVisitor, Visitor


def make_bytearray(buf):
    # Make a byte array constant from *buf*.
    b = bytearray(buf)
    n = len(b)
    return ir.Constant(ir.ArrayType(ir.IntType(8), n), b)


class LLVMFunctionVisitor(BlockVisitor):
    def __init__(self, module):
        self.module = module
        self.func = None
        self.builder = None
        self.loc = {}
        self.const = {}
        self.params = {}
        self.buildBlocks = True

    def _get_block(self, label):
        for block in self.func.blocks:
            if label == block.name:
                return block

    def _get_llvm_type(self, type):
        if type == "void":
            return ir.VoidType()
        elif type == "int":
            return ir.IntType(32)
        else:
            return ir.VoidType()  # TODO: treat other types

    def _extract_operation(self, inst):
        _modifier = {}
        _ctype = None
        _aux = inst.split("_")
        _opcode = _aux[0]
        if _opcode not in {"fptosi", "sitofp", "jump", "cbranch"} and len(_aux) > 1:
            _ctype = _aux[1]
            for i, _val in enumerate(_aux[2:]):
                if _val.isdigit():
                    _modifier["dim" + str(i)] = _val
                elif _val == "*":
                    _modifier["ptr" + str(i)] = _val
        return _opcode, _ctype, _modifier

    def _get_loc(self, target):
        try:
            if target[0] == "%":
                return self.loc[target]
            elif target[0] == "@":
                return self.module.get_global(target[1:])
        except KeyError:
            return None

    def _global_constant(self, builder_or_module, name, value, linkage="internal"):
        # Get or create a (LLVM module-)global constant with *name* or *value*.
        if isinstance(builder_or_module, ir.Module):
            mod = builder_or_module
        else:
            mod = builder_or_module.module
        data = ir.GlobalVariable(mod, value.type, name=name)
        data.linkage = linkage
        data.global_constant = True
        data.initializer = value
        data.align = 1
        return data

    def _cio(self, fname, format, *target):
        # Make global constant for string format
        mod = self.builder.module
        fmt_bytes = make_bytearray((format + "\00").encode("ascii"))
        global_fmt = self._global_constant(mod, mod.get_unique_name(".fmt"), fmt_bytes)
        fn = mod.get_global(fname)
        ptr_fmt = self.builder.bitcast(global_fmt, ir.IntType(8).as_pointer())
        return self.builder.call(fn, [ptr_fmt] + list(target))

    def _build_print(self, val_type, *args):
        """Build a print instruction, handling both value printing and newline"""
        # If we have arguments, we're printing a value
        if args:
            target = args[0]  # First argument is the target
            # get the object assigned to target
            _value = self._get_loc(target)
            if val_type == "int":
                self._cio("printf", "%d", _value)
            elif val_type == "float":
                self._cio("printf", "%.2f", _value)
            elif val_type == "char":
                self._cio("printf", "%c", _value)
            elif val_type == "string":
                self._cio("printf", "%s", _value)
        else:
            # No arguments means print newline
            self._cio("printf", "\n")

    def _build_alloc(self, val_type, target, **kwargs):
        """Build an alloc instruction, handling both simple variables and arrays"""
        # Get the base type
        base_type = self._get_llvm_type(val_type)
        
        # Check if we're allocating an array
        if 'dim0' in kwargs:
            # Create array type with the specified dimension
            array_size = int(kwargs['dim0'])
            array_type = ir.ArrayType(base_type, array_size)
            # Allocate the array
            self.loc[target] = self.builder.alloca(array_type, size=None, name=target[1:])
        else:
            # Simple variable allocation
            self.loc[target] = self.builder.alloca(base_type, size=None, name=target[1:])

    def _build_param(self, typ, target):
        self.params[target] = typ

    def _build_call(self, typ, name, target):
        """Build a function call instruction, handling both local and global functions"""
        # Get function reference
        func = self._get_loc(name)
        if func is None:
            # If function not in loc dict, try getting it directly from module
            func = self.module.get_global(name[1:])  # Remove the @ from name
            if func is None:
                raise RuntimeError(f"Function {name} not found")
        
        # Build argument list
        args = []
        for param in self.params:
            param_val = self._get_loc(param)
            if param in self.const:
                # Handle constant parameters
                param_val = ir.Constant(ir.IntType(32), self.const[param])
            args.append(param_val)
        
        # Make the call and store result
        self.loc[target] = self.builder.call(func, args, name=target[1:])
        self.params = {}

    def _build_jump(self, typ, target):
        self.builder.branch(self._get_block(target[1:]))

    def _build_cbranch(self, typ, cond, btrue, bfalse):
        """Build a conditional branch instruction, ensuring condition is boolean"""
        cond_val = self._get_loc(cond)
        
        # Convert condition to boolean (i1) if it's not already
        if isinstance(cond_val.type, ir.IntType) and cond_val.type.width != 1:
            # Compare with 0 to get boolean
            cond_val = self.builder.icmp_signed('!=', cond_val, ir.Constant(ir.IntType(32), 0))
        
        # Create conditional branch
        self.builder.cbranch(
            cond_val,
            self._get_block(btrue[1:]), 
            self._get_block(bfalse[1:])
        )

    def _build_return(self, val_type, target=None):
        """Build a return instruction, handling both void returns and returns with values"""
        if val_type == "void" or target is None:
            self.builder.ret_void()
        else:
            self.builder.ret(self._get_loc(target))

    def _build_literal(self, typ, mod, target):
        self.loc[target] = mod
        self.const[target] = mod

    def _build_load(self, typ, mod, target, **kwargs):
        """Build a load instruction"""
        # Handle constants first
        if mod in self.const:
            # Create a constant value
            value = ir.Constant(ir.IntType(32), self.const[mod])  # TODO: fix hard coded int type
            # Store it in loc for future use
            self.loc[target] = value
            return

        # Get the source location for non-constants
        source_loc = self._get_loc(mod)
        
        # Ensure we're loading from a pointer
        if not isinstance(source_loc.type, ir.PointerType):
            # If it's not a pointer, create a pointer by allocating and storing
            ptr = self.builder.alloca(source_loc.type)
            self.builder.store(source_loc, ptr)
            source_loc = ptr
        
        # Now we can safely load
        self.loc[target] = self.builder.load(source_loc)

    def _build_add(self, typ, val1, val2, target):
        loc_val1 = self._get_loc(val1)
        loc_val2 = self._get_loc(val2)

        if val1 in self.const:
            loc_val1 = ir.Constant(
                ir.IntType(32), self.const[val1]
            )  # TODO: fix hard coded int type
        if val2 in self.const:
            loc_val2 = ir.Constant(
                ir.IntType(32), self.const[val2]
            )  # TODO: fix hard coded int type

        self.loc[target] = self.builder.add(loc_val1, loc_val2, name=target[1:])

    def _build_sub(self, typ, val1, val2, target):
        loc_val1 = self._get_loc(val1)
        loc_val2 = self._get_loc(val2)

        if val1 in self.const:
            loc_val1 = ir.Constant(
                ir.IntType(32), self.const[val1]
            )  # TODO: fix hard coded int type
        if val2 in self.const:
            loc_val2 = ir.Constant(
                ir.IntType(32), self.const[val2]
            )  # TODO: fix hard coded int type

        self.loc[target] = self.builder.sub(loc_val1, loc_val2, name=target[1:])

    def _build_not(self, typ, val1, target):
        loc_val1 = self._get_loc(val1)

        if val1 in self.const:
            loc_val1 = ir.Constant(
                ir.IntType(32), self.const[val1]
            )  # TODO: fix hard coded int type

        # self.loc[target] =  self.builder.add(loc_val1, ir.Constant(ir.IntType(1), 1), name=target[1:])
        self.loc[target] = self.builder.icmp_signed(
            "==", loc_val1, ir.Constant(ir.IntType(1), 0), name=target[1:]
        )

    def _build_mul(self, typ, val1, val2, target):
        loc_val1 = self._get_loc(val1)
        loc_val2 = self._get_loc(val2)

        if val1 in self.const:
            loc_val1 = ir.Constant(
                ir.IntType(32), self.const[val1]
            )  # TODO: fix hard coded int type
        if val2 in self.const:
            loc_val2 = ir.Constant(
                ir.IntType(32), self.const[val2]
            )  # TODO: fix hard coded int type

        self.loc[target] = self.builder.mul(loc_val1, loc_val2, name=target[1:])

    def _build_mod(self, typ, val1, val2, target):
        loc_val1 = self._get_loc(val1)
        loc_val2 = self._get_loc(val2)

        if val1 in self.const:
            loc_val1 = ir.Constant(
                ir.IntType(32), self.const[val1]
            )  # TODO: fix hard coded int type
        if val2 in self.const:
            loc_val2 = ir.Constant(
                ir.IntType(32), self.const[val2]
            )  # TODO: fix hard coded int type

        self.loc[target] = self.builder.srem(loc_val1, loc_val2, name=target[1:])

    def _build_div(self, typ, val1, val2, target):
        loc_val1 = self._get_loc(val1)
        loc_val2 = self._get_loc(val2)

        if val1 in self.const:
            loc_val1 = ir.Constant(
                ir.IntType(32), self.const[val1]
            )  # TODO: fix hard coded int type
        if val2 in self.const:
            loc_val2 = ir.Constant(
                ir.IntType(32), self.const[val2]
            )  # TODO: fix hard coded int type

        self.loc[target] = self.builder.sdiv(loc_val1, loc_val2, name=target[1:])

    def _build_le(self, typ, val1, val2, target):
        loc_val1 = self._get_loc(val1)
        loc_val2 = self._get_loc(val2)

        if val1 in self.const:
            loc_val1 = ir.Constant(
                ir.IntType(32), self.const[val1]
            )  # TODO: fix hard coded int type
        if val2 in self.const:
            loc_val2 = ir.Constant(
                ir.IntType(32), self.const[val2]
            )  # TODO: fix hard coded int type

        self.loc[target] = self.builder.icmp_signed(
            "<=", loc_val1, loc_val2, name=target[1:]
        )

    def _build_lt(self, typ, val1, val2, target):
        loc_val1 = self._get_loc(val1)
        loc_val2 = self._get_loc(val2)

        if val1 in self.const:
            loc_val1 = ir.Constant(
                ir.IntType(32), self.const[val1]
            )  # TODO: fix hard coded int type
        if val2 in self.const:
            loc_val2 = ir.Constant(
                ir.IntType(32), self.const[val2]
            )  # TODO: fix hard coded int type

        self.loc[target] = self.builder.icmp_signed(
            "<", loc_val1, loc_val2, name=target[1:]
        )

    def _build_eq(self, typ, val1, val2, target):
        loc_val1 = self._get_loc(val1)
        loc_val2 = self._get_loc(val2)

        if val1 in self.const:
            loc_val1 = ir.Constant(
                ir.IntType(32), self.const[val1]
            )  # TODO: fix hard coded int type
        if val2 in self.const:
            loc_val2 = ir.Constant(
                ir.IntType(32), self.const[val2]
            )  # TODO: fix hard coded int type

        self.loc[target] = self.builder.icmp_signed(
            "==", loc_val1, loc_val2, name=target[1:]
        )

    def _build_ne(self, typ, val1, val2, target):
        loc_val1 = self._get_loc(val1)
        loc_val2 = self._get_loc(val2)

        if val1 in self.const:
            loc_val1 = ir.Constant(
                ir.IntType(32), self.const[val1]
            )  # TODO: fix hard coded int type
        if val2 in self.const:
            loc_val2 = ir.Constant(
                ir.IntType(32), self.const[val2]
            )  # TODO: fix hard coded int type

        self.loc[target] = self.builder.icmp_signed(
            "!=", loc_val1, loc_val2, name=target[1:]
        )

    def _build_gt(self, typ, val1, val2, target):
        loc_val1 = self._get_loc(val1)
        loc_val2 = self._get_loc(val2)

        if val1 in self.const:
            loc_val1 = ir.Constant(
                ir.IntType(32), self.const[val1]
            )  # TODO: fix hard coded int type
        if val2 in self.const:
            loc_val2 = ir.Constant(
                ir.IntType(32), self.const[val2]
            )  # TODO: fix hard coded int type

        self.loc[target] = self.builder.icmp_signed(
            ">", loc_val1, loc_val2, name=target[1:]
        )

    def _build_ge(self, typ, val1, val2, target):
        loc_val1 = self._get_loc(val1)
        loc_val2 = self._get_loc(val2)

        if val1 in self.const:
            loc_val1 = ir.Constant(
                ir.IntType(32), self.const[val1]
            )  # TODO: fix hard coded int type
        if val2 in self.const:
            loc_val2 = ir.Constant(
                ir.IntType(32), self.const[val2]
            )  # TODO: fix hard coded int type

        self.loc[target] = self.builder.icmp_signed(
            ">=", loc_val1, loc_val2, name=target[1:]
        )

    def _build_store(self, type, mod, target, **kwargs):
        """Build a store instruction, handling both simple variables and array/pointer operations"""
        # Get the value to store
        if mod in self.const:
            value = ir.Constant(ir.IntType(32), self.const[mod])  # TODO: fix hard coded int type
        else:
            value = self._get_loc(mod)
        
        # Get the target location
        target_loc = self._get_loc(target)
        
        # If we're storing to a pointer/array
        if 'ptr0' in kwargs:
            # Make sure we're storing to the right type
            if isinstance(target_loc.type, ir.PointerType):
                # If target is already a pointer, store directly
                self.builder.store(value, target_loc)
            else:
                # If not, we need to bitcast or handle the pointer type
                ptr_type = value.type.as_pointer()
                cast_ptr = self.builder.bitcast(target_loc, ptr_type)
                self.builder.store(value, cast_ptr)
        else:
            # Simple store operation
            self.builder.store(value, target_loc)

    def _build_elem(self, typ, val1, val2, target):
        """Build an element access instruction for arrays"""
        base_ptr = self._get_loc(val1)
        index = self._get_loc(val2)
        
        if val2 in self.const:
            index = ir.Constant(ir.IntType(32), self.const[val2])
        
        # Create a GEP (GetElementPtr) instruction
        zero = ir.Constant(ir.IntType(32), 0)
        ptr = self.builder.gep(base_ptr, [zero, index], name=target[1:] + "_ptr")
        self.loc[target] = ptr

    def _build_or(self, typ, val1, val2, target):
        """Build a logical OR operation"""
        loc_val1 = self._get_loc(val1)
        loc_val2 = self._get_loc(val2)

        # Handle constants
        if val1 in self.const:
            loc_val1 = ir.Constant(ir.IntType(32), self.const[val1])
        if val2 in self.const:
            loc_val2 = ir.Constant(ir.IntType(32), self.const[val2])

        # Convert to boolean values (non-zero is true)
        bool1 = self.builder.icmp_signed('!=', loc_val1, ir.Constant(ir.IntType(32), 0))
        bool2 = self.builder.icmp_signed('!=', loc_val2, ir.Constant(ir.IntType(32), 0))
        
        # Perform OR operation
        result = self.builder.or_(bool1, bool2)
        
        # Convert back to int (0 or 1)
        self.loc[target] = self.builder.zext(result, ir.IntType(32))

    def build(self, inst):
        opcode, ctype, modifier = self._extract_operation(inst[0])
        # Remove any trailing underscore from opcode
        opcode = opcode.rstrip('_')
        method_name = f"_build_{opcode}"
        if hasattr(self, method_name):
            getattr(self, method_name)(ctype, *inst[1:], **modifier)
        else:
            print(f"Warning: No {method_name}() method")

    def visit_BasicBlock(self, block):
        if self.buildBlocks:
            insts = block.instructions
            first_inst = None
            if len(insts) > 0 and len(insts[0]) > 0:
                first_inst = insts[0]
                first_op, first_ty, _ = self._extract_operation(first_inst[0])

                if first_inst is not None and first_op == "define":
                    # Check if function already exists before creating it
                    func_name = first_inst[1][1:]  # Remove the @ from name
                    existing_func = self.module.get_global(func_name)
                    if existing_func is None:
                        define_type = self._get_llvm_type(first_ty)
                        func_args = []
                        for arg in first_inst[2]:
                            func_args.append(self._get_llvm_type(arg[0]))
                        fnty = ir.FunctionType(define_type, tuple(func_args))
                        self.func = ir.Function(self.module, fnty, name=func_name)

                        for arg_inst, fun_arg in zip(first_inst[2], self.func.args):
                            self.loc[arg_inst[1]] = fun_arg
                    else:
                        self.func = existing_func
                        # Still need to set up argument mappings
                        for arg_inst, fun_arg in zip(first_inst[2], self.func.args):
                            self.loc[arg_inst[1]] = fun_arg

            self.func.append_basic_block(name=block.label)
        else:
            # get block
            self.builder = ir.IRBuilder(self._get_block(block.label))

            for inst in block.instructions:
                if "define" not in inst[0] and ":" not in inst[0]:
                    self.build(inst)

    def visit_ConditionBlock(self, block):
        if self.buildBlocks:
            insts = block.instructions
            first_inst = None
            if len(insts) > 0 and len(insts[0]) > 0:
                first_inst = insts[0]
                first_op, first_ty, _ = self._extract_operation(first_inst[0])

                if first_inst is not None and first_op == "define":
                    # Check if function already exists before creating it
                    func_name = first_inst[1][1:]  # Remove the @ from name
                    existing_func = self.module.get_global(func_name)
                    if existing_func is None:
                        define_type = self._get_llvm_type(first_ty)
                        func_args = []
                        for arg in first_inst[2]:
                            func_args.append(self._get_llvm_type(arg[0]))
                        fnty = ir.FunctionType(define_type, tuple(func_args))
                        self.func = ir.Function(self.module, fnty, name=func_name)

                        for arg_inst, fun_arg in zip(first_inst[2], self.func.args):
                            self.loc[arg_inst[1]] = fun_arg
                    else:
                        self.func = existing_func
                        # Still need to set up argument mappings
                        for arg_inst, fun_arg in zip(first_inst[2], self.func.args):
                            self.loc[arg_inst[1]] = fun_arg

            self.func.append_basic_block(name=block.label)
        else:
            # get block
            self.builder = ir.IRBuilder(self._get_block(block.label))

            for inst in block.instructions:
                if "define" not in inst[0] and ":" not in inst[0]:
                    self.build(inst)


class LLVMCodeGenerator(NodeVisitor):
    def __init__(self, viewcfg):
        self.viewcfg = viewcfg
        self.binding = binding
        self.binding.initialize()
        self.binding.initialize_native_target()
        self.binding.initialize_native_asmprinter()

        self.module = ir.Module(name=__file__)
        self.module.triple = self.binding.get_default_triple()

        self.engine = self._create_execution_engine()

        # declare external functions
        self._declare_printf_function()
        self._declare_scanf_function()

    def _create_execution_engine(self):
        """
        Create an ExecutionEngine suitable for JIT code generation on
        the host CPU.  The engine is reusable for an arbitrary number of
        modules.
        """
        target = self.binding.Target.from_default_triple()
        target_machine = target.create_target_machine()
        # And an execution engine with an empty backing module
        backing_mod = binding.parse_assembly("")
        return binding.create_mcjit_compiler(backing_mod, target_machine)

    def _declare_printf_function(self):
        voidptr_ty = ir.IntType(8).as_pointer()
        printf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
        printf = ir.Function(self.module, printf_ty, name="printf")
        self.printf = printf

    def _declare_scanf_function(self):
        voidptr_ty = ir.IntType(8).as_pointer()
        scanf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
        scanf = ir.Function(self.module, scanf_ty, name="scanf")
        self.scanf = scanf

    def _compile_ir(self):
        """
        Compile the LLVM IR string with the given engine.
        The compiled module object is returned.
        """
        # Create a LLVM module object from the IR
        llvm_ir = str(self.module)
        mod = self.binding.parse_assembly(llvm_ir)
        mod.verify()
        # Now add the module and make sure it is ready for execution
        self.engine.add_module(mod)
        self.engine.finalize_object()
        self.engine.run_static_constructors()
        return mod

    def save_ir(self, output_file):
        output_file.write(str(self.module))

    def execute_ir(self, opt, opt_file):
        mod = self._compile_ir()

        if opt:
            # apply some optimization passes on module
            pmb = self.binding.create_pass_manager_builder()
            pm = self.binding.create_module_pass_manager()

            pmb.opt_level = 0
            if opt == "ctm" or opt == "all":
                # Sparse conditional constant propagation and merging
                pm.add_sccp_pass()
                # Merges duplicate global constants together
                pm.add_constant_merge_pass()
                # Combine inst to form fewer, simple inst
                # This pass also does algebraic simplification
                pm.add_instruction_combining_pass()
            if opt == "dce" or opt == "all":
                pm.add_dead_code_elimination_pass()
            if opt == "cfg" or opt == "all":
                # Performs dead code elimination and basic block merging
                pm.add_cfg_simplification_pass()

            pmb.populate(pm)
            pm.run(mod)
            opt_file.write(str(mod))

        # Obtain a pointer to the compiled 'main' - it's the address of its JITed code in memory.
        main_ptr = self.engine.get_function_address("main")
        # To convert an address to an actual callable thing we have to use
        # CFUNCTYPE, and specify the arguments & return type.
        main_function = CFUNCTYPE(c_int)(main_ptr)
        # Now 'main_function' is an actual callable we can invoke
        res = main_function()

    def _global_constant(self, builder_or_module, name, value, linkage=""):
        # Get or create a (LLVM module-)global constant with *name* or *value*.
        if isinstance(builder_or_module, ir.Module):
            mod = builder_or_module
        else:
            mod = builder_or_module.module
        data = ir.GlobalVariable(mod, value.type, name=name)
        data.linkage = linkage
        data.global_constant = True
        data.initializer = value
        data.align = 1
        return data

    def _generate_global_instructions(self, node):
        for const in node:
            types = const[0].split("_")
            if len(types) <= 2:
                typ = const[0].split("_")[1]
            else:
                typ = "array"
                size = types[2]

            name = const[1][1:]
            value = const[2]
            if typ == "string":
                const_bytes = make_bytearray((value + "\00").encode("ascii"))
                self._global_constant(self.module, name, const_bytes)
            elif typ == "int":
                self._global_constant(
                    self.module, name, ir.Constant(ir.IntType(32), value)
                )
            elif typ == "array":
                self._global_constant(
                    self.module,
                    name,
                    ir.Constant(ir.ArrayType(ir.IntType(32), int(size)), value),
                )
            else:
                print("Type not yet supported")

    def visit_Program(self, node):
        # First pass: Register all function declarations
        for _decl in node.gdecls:
            if isinstance(_decl, FuncDef):
                # Create basic function declaration
                bb = LLVMFunctionVisitor(self.module)
                bb.buildBlocks = True
                # Get the first instruction directly from the cfg (which is a BasicBlock)
                first_inst = _decl.cfg.instructions[0]
                first_op, first_ty, _ = bb._extract_operation(first_inst[0])
                if first_op == "define":
                    define_type = bb._get_llvm_type(first_ty)
                    func_args = []
                    for arg in first_inst[2]:
                        func_args.append(bb._get_llvm_type(arg[0]))
                    fnty = ir.FunctionType(define_type, tuple(func_args))
                    bb.func = ir.Function(self.module, fnty, name=first_inst[1][1:])

        # Generate global instructions
        self._generate_global_instructions(node.text)
        
        # Second pass: Generate function bodies
        for _decl in node.gdecls:
            if isinstance(_decl, FuncDef):
                bb = LLVMFunctionVisitor(self.module)
                # Visit the CFG to define the Basic Blocks
                bb.visit(_decl.cfg)
                # Visit CFG again to create the instructions inside Basic Blocks
                bb.buildBlocks = False
                bb.visit(_decl.cfg)
                if self.viewcfg:
                    dot = binding.get_function_cfg(bb.func)
                    gv = binding.view_dot_graph(dot, _decl.decl.declname.name, False)
                    gv.filename = _decl.decl.declname.name + ".ll.gv"
                    gv.view()


if __name__ == "__main__":

    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        help="Path to file to be used to generate LLVM IR. By default, this script runs the LLVM IR without any optimizations.",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--cfg",
        help="show the CFG of the optimized uCIR for each function in pdf format",
        action="store_true",
    )
    parser.add_argument(
        "--llvm-opt",
        default=None,
        choices=["ctm", "dce", "cfg", "all"],
        help="specify which llvm pass optimizations should be enabled",
    )
    parser.add_argument(
        "--save-ir",
        help="save the generated LLVM IR to the specified file",
        type=str,
        metavar="FILE"
    )
    parser.add_argument(
        "--save-ast",
        help="save the AST to the specified file",
        type=str,
        metavar="FILE"
    )
    args = parser.parse_args()

    create_cfg = args.cfg
    llvm_opt = args.llvm_opt

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

    # Save AST if requested
    if args.save_ast:
        with open(args.save_ast, 'w') as f:
            ast.show(buf=f, showcoord=True)

    sema = Visitor()
    sema.visit(ast)

    gen = CodeGenerator(False)
    gen.visit(ast)

    llvm = LLVMCodeGenerator(create_cfg)
    llvm.visit(ast)
    if args.save_ir:
        with open(args.save_ir, 'w') as f:
            llvm.save_ir(f)
    llvm.execute_ir(llvm_opt, None)

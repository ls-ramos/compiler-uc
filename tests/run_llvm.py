import argparse
from uc.uc_analysis import DataFlow
from uc.uc_code import CodeGenerator
from uc.uc_interpreter import Interpreter
from uc.uc_llvm import LLVMCodeGenerator
from uc.uc_parser import UCParser
from uc.uc_sema import Visitor


def run_llvm(input_path):
    p = UCParser(debug=False)
    with open(input_path) as f_in:
        ast = p.parse(f_in.read())
        sema = Visitor()
        sema.visit(ast)
        gen = CodeGenerator(False)
        gen.visit(ast)
        llvm = LLVMCodeGenerator(False)
        llvm.visit(ast)
        llvm.execute_ir(None, None)


if __name__ == "__main__":

    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        help="Path to file to be used to generate LLVM IR.",
        type=str,
    )
    args = parser.parse_args()
    run_llvm(args.input_file)

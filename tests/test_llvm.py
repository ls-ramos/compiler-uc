import io
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import pytest
from uc.uc_analysis import DataFlow
from uc.uc_code import CodeGenerator
from uc.uc_interpreter import Interpreter
from uc.uc_llvm import LLVMCodeGenerator
from uc.uc_parser import UCParser
from uc.uc_sema import Visitor
import sys

name = [
    "t01",
    "t02",
    "t03",
    "t04",
    "t05",
    "t06",
    "t07",
    "t08",
    "t09",
    "t10",
    "t11",
    "t12",
    "t13",
    "t14",
    "t15",
    "t16",
    "t17",
    "t18",
    "t19",
    "t20",
    "t21",
    "t22",
    "t23",
    "t24",
    "t25",
]


def resolve_test_files(test_name):
    input_file = test_name + ".in"
    expected_file = test_name + ".out"

    # get current dir
    current_dir = Path(__file__).parent.absolute()

    # get absolute path to inputs folder
    test_folder = current_dir / Path("in-out")

    # get input path and check if exists
    input_path = test_folder / Path(input_file)
    assert input_path.exists()

    # get expected test file real path
    expected_path = test_folder / Path(expected_file)
    assert expected_path.exists()

    run_llvm_script_path = current_dir / Path("run_llvm.py")
    assert run_llvm_script_path.exists()

    return input_path, expected_path, run_llvm_script_path


@pytest.mark.parametrize("test_name", name)
# capfd will capture the stdout/stderr outputs generated during the test
def test_llvm(test_name, capsys):
    input_path, expected_path, run_llvm_script_path = resolve_test_files(test_name)

    res = subprocess.run(
        [sys.executable, run_llvm_script_path, input_path], capture_output=True
    )
    res_stdout = res.stdout.decode("utf-8")
    res_stderr = res.stderr.decode("utf-8")
    with open(expected_path) as f_ex:
        expect = f_ex.read()
    assert res_stdout == expect
    assert res_stderr == ""

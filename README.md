# uC Compiler - LLVM IR Generator

A compiler for the uC (micro C) language that generates LLVM IR code. This project is part of the MC921 - Compiler Construction course at Unicamp.

uC is a simplified version of C, with a subset of the language features.

uC follows the following specifications:

- [BNF grammar specification](https://github.com/ls-ramos/compiler-uc/blob/main/docs/bnf_grammar.md)
- [Semantic rules](https://github.com/ls-ramos/compiler-uc/blob/main/docs/semantic_rules.md)

## About

This is a compiler that takes uC (a simplified version of C) source code as input and generates LLVM IR (Intermediate Representation) code that can be executed. It includes:

- Lexical analysis
- Parsing
- Semantic analysis 
- LLVM IR code generation

## Quick Start

1. Install requirements:
```sh
python3 -m pip install -r requirements.txt
```

2. Run the compiler:
```sh
python3 uc/uc_llvm.py ./tests/in-out/t01.in
```

Or use the shortcut script:
```sh
./ucc input_file.uc
```

For more options:
```sh
python3 uc/uc_llvm.py -h
```

## Running Tests

1. Setup the environment:
```sh
source setup.sh
```

2. Run the tests:
```sh
pytest tests/test_llvm.py
```

To run specific tests you can use:
```sh
pytest tests/test_llvm.py -k t01
```
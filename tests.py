from dataclasses import field
from typing import Dict, List

from mltoolkit import argclass, parse_args, GeneralArguments


@argclass
class A:
    one: int = field(default=1)
    two: int = field(default=2)


@argclass
class B(GeneralArguments):
    dict_of_As: Dict[str, A] = field(default_factory=dict)
    list_of_As: List[A] = field(default_factory=list)


def main():
    args = parse_args(A)
    print()


main()

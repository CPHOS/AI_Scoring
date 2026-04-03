#!/usr/bin/env python3
"""CLI 入口点."""

from src.recognize.service import cli_main


def main() -> int:
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())

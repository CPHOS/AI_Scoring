#!/usr/bin/env python3
"""CLI 入口点."""

import sys


def main() -> int:
    # Subcommand routing: judge / direct / recognize (default)
    if len(sys.argv) >= 2 and sys.argv[1] == "judge":
        from src.judge.service import judge_cli_main
        return judge_cli_main(sys.argv[2:])

    if len(sys.argv) >= 2 and sys.argv[1] == "direct":
        from src.direct_vlm.service import direct_vlm_cli_main
        return direct_vlm_cli_main(sys.argv[2:])

    # 'recognize' explicit subcommand — strip it and fall through
    if len(sys.argv) >= 2 and sys.argv[1] == "recognize":
        sys.argv = [sys.argv[0]] + sys.argv[2:]

    from src.recognize.service import cli_main
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())

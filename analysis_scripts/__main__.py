# mypackage/__main__.py
import argparse
from . import wrapper
#from .module2 import function2

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Subparser for command1
    parser_command1 = subparsers.add_parser('eval_vol')
    # Add arguments for command1 if needed
    wrapper.eval_vol.add_args(parser_command1)
    parser_command1.set_defaults(func=wrapper.eval_vol.main)

    # Subparser for command2
    parser_command2 = subparsers.add_parser('analyze')
    # Add arguments for command2 if needed
    wrapper.analyze.add_args(parser_command2)
    parser_command2.set_defaults(func=wrapper.analyze.main)

    # Subparser for command2
    parser_command2 = subparsers.add_parser('parse_pose')
    # Add arguments for command2 if needed
    wrapper.parse_pose.add_args(parser_command2)
    parser_command2.set_defaults(func=wrapper.parse_pose.main)

    # Subparser for command2
    parser_command2 = subparsers.add_parser('prepare')
    # Add arguments for command2 if needed
    wrapper.prepare.add_args(parser_command2)
    parser_command2.set_defaults(func=wrapper.prepare.main)

    # Subparser for command2
    parser_command2 = subparsers.add_parser('prepare_multi')
    # Add arguments for command2 if needed
    wrapper.prepare_multi.add_args(parser_command2)
    parser_command2.set_defaults(func=wrapper.prepare_multi.main)

    args = parser.parse_args()
    if args.command:
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()


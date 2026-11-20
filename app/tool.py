import sys
from cytools_core.transports.cli import run_cli
from settings import DATA
from studio_runtime import create_runtime


if __name__ == '__main__':
    argv = sys.argv[1:]
    if not any(item == '--data-dir' or item.startswith('--data-dir=') for item in argv):
        argv = ['--data-dir', str(DATA / 'jobs'), *argv]
    raise SystemExit(run_cli(create_runtime, argv))

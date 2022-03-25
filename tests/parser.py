# python tests/parser.py -c configs/template.yml -o global.debug=False
from opt import Opts

if __name__ == "__main__":
    opts = Opts().parse_args()
    print(opts)

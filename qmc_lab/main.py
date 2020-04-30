import time
import sys
import os
import getopt
import numpy as np
import qmc
import wavefunction as wf

class ConfigFile:
    '''Parse configuration options for the run'''
    def __init__(self, file_name):
        self.file_name = file_name
        f = open(file_name, "r")
        for line in f:
            exec("self." + line.strip())
        f.close()

def read_input():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
    except (getopt.GetoptError, err):
        print(str(err), file=sys.stderr)
        usage()
        sys.exit(2)
    for o, a in opts:
        if o in ("-h", "--help"):
            print("\nUsage: python main.py input_file > output_file", \
                  file=sys.stderr)
            sys.exit()
        else:
            assert False, "unhandled option"
    filename = args[0]
    return ConfigFile(filename)

def build_wf(config):
    n_orb, spin = config.n_orb, config.spin
    if n_orb == 1 and spin == 0:
        slater = wf.Slater1Orb(config.zeta)
    elif n_orb == 2 and spin == 0:
        slater = wf.Symm2Orb(config.Z, config.zeta, config.zeta1, config.zeta2)
    elif n_orb == 2 and spin == 1:
        slater = wf.Antisymm2Orb(config.Z, config.zeta, config.zeta1, \
                                 config.zeta2, config.tau)
    else:
        raise ValueError('Error: Unimplemented spin/n_orb combination in input.')
    jas = wf.Jastrow(config.b1, config.b2)
    return wf.getWavefunction('SlaterJastrow', slater, jas)

def run_qmc(config):
    #construct wf
    wavefunction = build_wf(config)
    #run qmc
    qmc.QMC(wavefunction, config).run()
    return

if __name__ == "__main__":
    np.random.seed(1)
    config = read_input()
    start_time = time.time()
    run_qmc(config)
    end_time = time.time()
    print("Time: %.2fs" % (end_time - start_time))

import numpy as np
import re
import math
from observables import observables_dict
from copy import deepcopy

class Walker:
    def __init__(self, wf, coord = None, weight = 1.):
        self.wf = wf
        self.coord = np.random.randn(2, 3) if (coord is None) else coord
        self.wf_amp, self.laplacian, self.drift_velocity = wf(self.coord)
        self.weight = weight

    def take_step(self, tau):
        # The * converts tuple (2 3) to 2 integers
        "XXX - Eq. 16"
        return Walker(self.wf, proposed_coord, self.weight)

    def propagate_to(self, new_coord, tau):
        #Amplitude (w/o normalization) to propagate to new_coord in time tau
        "XXX - Eq. 14"
        return prop
    
class QMC:
    def __init__(self, wavefunction, config):
        self.method = config.run_type
        self.n_equil = config.n_step_equil
        self.n_step = config.n_step_per_block
        self.n_block = config.n_block
        self.wavefunction = wavefunction
        self.Z = config.Z
        self.tau = config.tau
        self.w_targ = config.weight_target
        self.n_gen_reset = config.n_generation_reset_weight
        self.w_max = 2
        self.w_min = 0.5

        self.observables = observables_dict(self.Z)
        self.v_sum = {} #cumulative weighted sum
        self.v_sq_sum = {} #cumulative weighted sum of squares
        self.vblk_sum = {} #sum in a single block
        self.vblk_sq_sum = {} #cumulative weighted sum of squares

        self.walkers = [Walker(self.wavefunction) for n in range(int(self.w_targ))]
        self.w_sum = 0.
        self.w_sq_sum = 0.
        self.wblk_sum = 0.
        self.wblk_sq_sum = 0.
        self.e_trial = np.average([self.observables['energy'](walker) \
                                   for walker in self.walkers])

        self.count_total = 0
        self.count_accept = 0

    #Propagator methods
    def reweight(self, new_wkr, old_wkr):
        """Update weights of new_wkr"""
        "XXX - Eq. 26"
        return

    def accept(self, new_wkr, old_wkr):
        """Return True if new_wkr is accepted, otherwise False"""
        "XXX - Eq. 15"
        return new_wkr_accepted

    #Accumulate methods
    def clear(self):
        for key, obs in self.observables.items():
            self.v_sum[key] = 0.
            self.v_sq_sum[key] = 0.
            self.vblk_sum[key] = 0.
            self.vblk_sq_sum[key] = 0.
        self.w_sum = 0.
        self.w_sq_sum = 0.
        self.wblk_sum = 0.
        self.wblk_sq_sum = 0.
        self.count_total = 0
        self.count_accept = 0
        return

    def log_observables(self, walker):
        weight = walker.weight
        self.wblk_sum += weight
        self.w_sq_sum += weight**2
        for key, obs in self.observables.items():
            val = obs(walker)
            self.vblk_sum[key] += weight*val
            self.v_sq_sum[key] += weight*(val**2)
        return

    def average(self, key):
        return self.v_sum[key]/self.w_sum

    def sigma(self, key):
        avg = self.v_sum[key]/self.w_sum
        n_eff = self.w_sum**2/self.w_sq_sum
        n_block_eff = self.w_sum**2/self.wblk_sq_sum
        if n_eff > 1:
            pref1 = n_eff/(n_eff - 1)
            sigma = np.sqrt(pref1*(self.v_sq_sum[key]/self.w_sum - avg**2))
        else:
            sigma = np.inf
        if n_block_eff > 1:
            pref2 = n_block_eff/(n_block_eff-1)
            block_sigma = np.sqrt(pref2*(self.vblk_sq_sum[key]/self.w_sum - avg**2))
        else:
            block_sigma = np.inf
        return sigma, block_sigma

    def end_block(self):
        self.w_sum += self.wblk_sum
        self.wblk_sq_sum += self.wblk_sum**2
        for key, obs in self.observables.items():
            self.v_sum[key] += self.vblk_sum[key]
            self.vblk_sum[key] /= self.wblk_sum
            self.vblk_sq_sum[key] += self.wblk_sum*(self.vblk_sum[key])**2
            self.vblk_sum[key] = 0
        self.wblk_sum = 0
        print()
        #Print error for observables
        for key in self.observables.keys():
            #Avg
            avg = self.average(key)
            sigma, block_sigma = self.sigma(key)
            n_block_eff = self.w_sum**2/self.wblk_sq_sum
            block_err = block_sigma/np.sqrt(n_block_eff)
            print(key, ": %.6f (%.6f)" % (avg, block_err), flush=True)
        return

    def energy_estimator(self):
        e_prev = self.v_sum["energy"] #energy from previous blocks
        e_blk = self.vblk_sum["energy"] #energy from this block
        e_tot = e_prev + e_blk
        w_tot = self.wblk_sum + self.w_sum #previous weights + weights from this block
        return e_tot/w_tot

    def integerize_walkers(self):
        new_walkers = []
        w_total = sum([walker.weight for walker in self.walkers])
        for walker in self.walkers:
            if (w_total > 2):
                n_copy = int(walker.weight + np.random.random())
                walker.weight = 1.0
            else:
                n_copy = 1
            for i in range(n_copy):
                new_walkers.append(deepcopy(walker))
        self.walkers = new_walkers
        return

    def split_join(self):
        def split(walker):
            n_copy = int(walker.weight)
            walker.weight /= n_copy
            for n in range(n_copy):
                new_walkers.append(deepcopy(walker))

        def join(walker1, walker2):
            w1, w2 = walker1.weight, walker2.weight
            if np.random.rand() < w1/(w1 + w2):
                walker1.weight = w1 + w2
                new_walkers.append(walker1)
            else:
                walker2.weight = w1 + w2
                new_walkers.append(walker2)

        new_walkers = []
        unjoined = None
        for walker in self.walkers:
            if walker.weight > self.w_max:
                split(walker)
            elif walker.weight < self.w_min:
                if unjoined is None:
                    unjoined = walker
                else:
                    join(unjoined, walker)
                    unjoined = None
            else:
                new_walkers.append(walker)
        if unjoined is not None:
            new_walkers.append(unjoined)
        self.walkers = new_walkers
        return

    def print_results(self, name, avg, sigma, block_sigma, block_err, corr_time):
        #TODO: Change names/make cleaner
        temp_error = '%.1e' % block_err
        # The first two digits, e.g. if temp_error is 2.1e-03, it would be '21'
        final_error = temp_error[:3].replace('.', '')
        if block_err > 0.0:
            match = re.search('\d$',str(temp_error))
            # match.group() gives the exponent
            # e.g. if temp_error is 2e-03, it would be '3'
            n_digits = str(int(match.group()) + 1)
            format_average = '%.' + n_digits + 'f'
            # truncate self.average to the number of digits from match.group() + 1
            final_average = format_average % avg
        else:
            final_average = avg
        epsilon = 1.e-4
        if sigma < epsilon:
            res = ("%16s" % name + ': ' + str(final_average) + '('
                   + str(final_error) + '), Sigma: ' + ("%.3f" % sigma)
                   + ', Autocorrelation Time is not well defined')
        else:
            res = ("%16s" % name + ': ' + str(final_average) + '('
                   + str(final_error) + '), Sigma: ' + ("%.3f" % sigma)
                   + ', T_corr: ' + ("%.1f" % corr_time))
        print(res)
        return

    def finalize(self):
        n_block_eff = self.w_sum**2/self.wblk_sq_sum
        for key, obs in self.observables.items():
            avg = self.average(key)
            sigma, block_sigma = self.sigma(key)
            block_err = block_sigma/np.sqrt(n_block_eff)
            corr_time = self.n_step*self.w_targ*(block_sigma/sigma)**2
            self.print_results(key, avg, sigma, block_sigma, block_err, \
                               corr_time)
        return

    def accumulate(self, n_block, n_step, equil_mode = False):
        self.clear()
        for i in range(n_block):
            for j in range(n_step):
                for n, walker in enumerate(self.walkers):
                    self.count_total += 1
                    #update walker
                    proposed = walker.take_step(self.tau)
                    if self.accept(proposed, walker):
                        if self.method == 'dmc':
                            self.reweight(proposed, walker)
                        self.walkers[n] = proposed
                        self.count_accept += 1
                    #measure walker
                    self.log_observables(self.walkers[n])
                #adjust trial energy and do branching using either split-join or integerize
                if self.method == "dmc":
                    e_est = self.energy_estimator()
                    """Update e_trial"""
                    self.e_trial = "XXX - Eq. 27"
                    self.split_join()
                    #self.integerize_walkers()
            self.end_block()
        if not equil_mode:
            self.finalize()
        return

    def run(self):
        #equilibrate
        n_equil_blocks = 1
        self.accumulate(n_equil_blocks, self.n_equil, equil_mode = True)
        #Accumulate
        self.accumulate(self.n_block, self.n_step)
        acc_ratio = self.count_accept/self.count_total
        print("\nFor tau = %.2e, acceptance ratio = %.2f" % (self.tau, acc_ratio))
        return

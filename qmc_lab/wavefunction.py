import sys
import numpy as np
from numpy import sqrt, exp, sum, random
from numpy.linalg import norm

def getWavefunction(wavefunction_type, *args):
    if wavefunction_type == "SlaterJastrow":
        return SlaterJastrow(*args)
    else:
        print >> sys.stderr, "Wavefunction of type ", wavefunction_type, " is not implemented!"
        sys.exit(2)

class Wavefunction(object):
    '''
    Generic Wavefunction in a QMC calculation.
    '''
    def __init__(self):
        NotImplemented
    def value(self):
        NotImplemented
    def drift_velocity(self):
        NotImplemented
    def laplacian(self):
        NotImplemented

class Slater1Orb(Wavefunction):
  """ 
  Slater determinant, Det, for one up and one down electron reduces to product of orbitals.
  Here orbitals are chosen to be exp(-zeta*r).
  """
  def __init__(self, zeta=2):
    super(Wavefunction, self).__init__()
    self.zeta = zeta

  """Note: For the functions below, r is a 2x3 array. E.g. r[0] (r[1]) is the 3D array
    representing the position of the first (second) electron."""

  def value(self,r):
    """Return the value of the wavefunction at position r"""
    "XXX"
    return sla_val

  def drift_velocity(self,r):
    """Return the drift_velocty, grad(Det)/Det, at position r"""
    "XXX"
    return sla_grad

  def local_laplacian(self,r):
    """Return the local laplacian, Laplacian(Det)/Det, at position r"""
    "XXX"
    return sla_lap

class Symm2Orb(Wavefunction):
    """Slater determinant, Det, for two electrons (w/ antisymmetric spin part) occupying two orbitals"""
    def __init__(self, Z, zeta, zeta1, zeta2):
        super(Wavefunction, self).__init__()
        self.Z = Z
        self.zeta = zeta
        self.zeta1 = zeta1
        self.zeta2 = zeta2

    def value(self,r):
        """Return the value of the wavefunction at position r"""
        "XXX"
        return sla_val
    
    def drift_velocity(self,r):
        """Return the drift_velocty, grad(Det)/Det, at position r"""
        "XXX"
        return sla_grad

    def local_laplacian(self,r):
        """Return the local laplacian, Laplacian(Det)/Det, at position r"""
        "XXX"
        return sla_lap

class Antisymm2Orb(Wavefunction):
    """Slater determinant, Det, for two electrons (w/ symmetric spin part) occupying two orbitals"""
    def __init__(self, Z, zeta, zeta1, zeta2, tau):
        super(Wavefunction, self).__init__()
        self.Z = Z
        self.zeta = zeta
        self.zeta1 = zeta1
        self.zeta2 = zeta2
        self.tau = tau

    def value(self,r):
        """Return the value of the wavefunction at position r"""
        "XXX"
        return sla_val
    
    def drift_velocity(self,r):
        """Return the drift_velocty, grad(Det)/Det, at position r"""
        "XXX"
        av = 1 #Implement averaged drift velocity term
        return av*sla_grad

    def local_laplacian(self,r):
        """Return the local laplacian, Laplacian(Det)/Det, at position r"""
        "XXX"
        return sla_lap

#-------------------------

class Jastrow(Wavefunction):
    """
    Simple e-e Jastrow factor J_ee(r_12) = exp(f_ee(r_12)) = exp(b1*r_12/(1+b2*r_12))
    """
    def __init__(self, b1=0.5, b2=1.0):
        super(Wavefunction, self).__init__()
        self.b1 = b1
        self.b2 = b2

    def value(self,r):
        """Return the value of the jastrow at position r"""
        "XXX"
        return jas_val
    
    def drift_velocity(self,r):
        """Return the drift_velocty, grad(Jas)/Jas, at position r"""
        "XXX"
        av = 1 #Implement averaged drift velocity term
        return av*jas_grad

    def local_laplacian(self,r):
        """Return the local laplacian, Laplacian(Jas)/Jas, at position r"""
        "XXX"
        return jas_lap
  
#-------------------------

class SlaterJastrow(Wavefunction):
    """ SlaterJastrow = Slater * Jastrow """
    def __init__(self,wf1,wf2):
        super(Wavefunction, self).__init__()
        self.wf1 = wf1
        self.wf2 = wf2
  
    def value(self,r):
        return self.wf1.value(r)*self.wf2.value(r)
  
    def drift_velocity(self,r):
        sla_grad = self.wf1.drift_velocity(r)
        jas_grad = self.wf2.drift_velocity(r)
        sla_jas_grad = sla_grad + jas_grad
        return sla_jas_grad
  
    def local_laplacian(self,r):
        sla_grad = self.wf1.drift_velocity(r)
        jas_grad = self.wf2.drift_velocity(r)
        sla_lap = self.wf1.local_laplacian(r)
        jas_lap = self.wf2.local_laplacian(r)
        sla_jas_lap = sla_lap + jas_lap + 2*sum(sla_grad * jas_grad ,axis=1)
        return sla_jas_lap
  
    def __call__(self, r):
        return self.value(r), self.local_laplacian(r), self.drift_velocity(r)

#**********************************************************

def numeric_err(r,wf,delta = 1e-5):
    """ Test analytic Laplacian using numerical Laplacian """
    wf_mid = wf.value(r)
    grad_analyt = wf.drift_velocity(r)
    lap_analyt = wf.local_laplacian(r)
    nelec = r.shape[0]
    ndim = r.shape[1]
# nsample = r.shape[2]

    grad_numeric = np.zeros(grad_analyt.shape)
    lap_numeric = np.zeros(lap_analyt.shape)
    shift = np.zeros(r.shape)
    for p in range(nelec):
        for d in range(ndim):
          shift[p,d] += delta
          wf_plus = wf.value(r+shift)
          shift[p,d] -= 2*delta
          wf_minus = wf.value(r+shift)
          grad_numeric[p,d] = (wf_plus-wf_minus)/(2*wf_mid*delta)
          lap_numeric[p] += (wf_plus+wf_minus-2*wf_mid)/(wf_mid*delta**2)
          shift[p,d] = 0
    grad_err = sqrt(sum((grad_numeric-grad_analyt)**2)/(nelec*ndim))
    lap_err = sqrt(sum((lap_numeric-lap_analyt)**2)/(nelec))

    return grad_err, lap_err
#**********************************************************

def test_numeric_analyt(wf):
    df = {'delta':[],
        'gradient err':[],
        'laplacian err':[]
        }
    for delta in [1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]:
        grad_err, lap_err = numeric_err(r,wf,delta)
        df['delta'].append(delta)
        df['gradient err'].append(grad_err)
        df['laplacian err'].append(lap_err)
  
    df = pd.DataFrame(df)
    pd.options.display.float_format = '{:.2e}'.format
  # df.style.format({'delta': "{:.2e}",'gradient err': "{:.2e}",'laplacian err': "{:.2e}"})
    df.style.apply
    print(df)
    return df
#**********************************************************

if __name__ == "__main__":
    import pandas as pd
    random.seed(4671321)    # Set random seed so we get same result each time
  # r = random.randn(2,3,5) # 2 electrons, 3-dim, 5 samples
    r = random.randn(2,3) # 2 electrons, 3-dim
  # r = np.zeros(shape=(2,3,1), dtype=float)
  # r[0,0,0]=1.
  # r[1,0,0]=0.5
  # r = [[[1.][0.][0.]][[0.][0.][0.]]]
  # print('r',r)
  # print('r',type(r))
  
    print("Slater wavefunction")
    test_numeric_analyt(Slater1Orb(2.0))
  
    print("\nJastrow wavefunction")
    test_numeric_analyt(Jastrow(0.5,1.0))
  
    print("\nSlater-Jastrow wavefunction")
    test_numeric_analyt(SlaterJastrow(Slater1Orb(2.0),Jastrow(0.5,1.0)))
  
    print("\nSlater2Orb WF")
    test_numeric_analyt(SlaterJastrow(Slater2Orb(1, 1, 1.18, 0.55, 1), Jastrow(0.5, 0.27)))

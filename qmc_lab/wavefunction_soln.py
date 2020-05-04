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

  def value(self,r):
    r_en = np.sqrt(np.sum(r**2,axis=1))
    sla_val = exp(-self.zeta*r_en[0]) * exp(-self.zeta*r_en[1])
    return sla_val

  def drift_velocity(self,r):
# grad(Det)/Det
    r_en = np.sqrt(np.sum(r**2,axis=1))
    sla_grad = -self.zeta*r/r_en[:,np.newaxis]
    return sla_grad

  def local_laplacian(self,r):
# Laplacian(Det)/Det
    r_en = np.sqrt(np.sum(r**2,axis=1))
    sla_val = exp(-self.zeta*r_en[0]) * exp(-self.zeta*r_en[1])
    sla_grad = -self.zeta*r/r_en[:,np.newaxis]
    sla_lap  = self.zeta**2 -2*self.zeta/r_en
    return sla_lap

class Slater2Orb(Wavefunction):
    def __init__(self, Z, zeta, zeta1, zeta2, sign):
        super(Wavefunction, self).__init__()
        self.Z = Z
        self.zeta = zeta
        self.zeta1 = zeta1
        self.zeta2 = zeta2
        self.sign = sign

    def phi1(self, r):
        return np.exp(-self.zeta*norm(r))

    def dphi1(self, r):
        unit = r/norm(r)
        return -self.zeta*self.phi1(r)*unit

    def lphi1(self, r):
        return (self.zeta**2 - 2*self.zeta/norm(r))*self.phi1(r)

    def phi2(self, r):
        z1, z2 = self.zeta1, self.zeta2
        nr = norm(r)
        return np.exp(-z1*nr) + (z1 - self.Z)*nr*np.exp(-z2*nr)

    def dphi2(self, r):
        z1, z2 = self.zeta1, self.zeta2
        nr = norm(r)
        unit = r/nr
        t1 = -z1*np.exp(-z1*nr)*unit
        t2 = (z1 - self.Z)*nr*np.exp(-z2*nr)*(1/nr - z2)*unit
        return t1 + t2

    def lphi2(self, r):
        z1, z2, nr = self.zeta1, self.zeta2, norm(r)
        lap = 0
        lap += (z1**2 - 2*z1/nr)*np.exp(-z1*nr)
        lap += (2/nr**2 - 4*z2/nr + z2**2)*(z1 - self.Z)*nr*np.exp(-z2*nr)
        return lap

    def value(self, r):
        p1, p2 = self.phi1, self.phi2
        r1, r2 = r
        return p1(r1)*p2(r2) + self.sign*p1(r2)*p2(r1)
    
    def drift_velocity(self, r):
        p1, p2 = self.phi1, self.phi2
        d1, d2 = self.dphi1, self.dphi2
        r1, r2 = r
        dv = np.zeros((2, 3))
        dv[0] += d1(r1)*p2(r2) + self.sign*p1(r2)*d2(r1)
        dv[1] += p1(r1)*d2(r2) + self.sign*d1(r2)*p2(r1)
        return dv/self.value(r)

    def local_laplacian(self, r):
        p1, p2 = self.phi1, self.phi2
        l1, l2 = self.lphi1, self.lphi2
        r1, r2 = r
        lap = np.zeros(2)
        lap[0] += l1(r1)*p2(r2) + self.sign*p1(r2)*l2(r1)
        lap[1] += p1(r1)*l2(r2) + self.sign*l1(r2)*p2(r1)
        return lap/self.value(r)

class Symm2Orb(Slater2Orb):
    def __init__(self, Z, zeta, zeta1, zeta2):
        Slater2Orb.__init__(self, Z, zeta, zeta1, zeta2, 1)

class Antisymm2Orb(Slater2Orb):
    def __init__(self, Z, zeta, zeta1, zeta2, tau):
        Slater2Orb.__init__(self, Z, zeta, zeta1, zeta2, -1)
        self.tau = tau

    def drift_velocity(self, r):
        dv = Slater2Orb.drift_velocity(self, r)
        vn = norm(dv)
        av = (-1 + np.sqrt(1 + 2*vn**2*self.tau))/(vn**2*self.tau)
        return av*dv
         
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
    r_ee = sqrt(sum((r[0,:]-r[1,:])**2,axis=0)) # sum over x,y,z (middle index is now axis = 0)
    jas_val = exp(self.b1*r_ee/(1+self.b2*r_ee))
    return jas_val

  def drift_velocity(self,r):
    r_ee = (sum((r[0,:]-r[1,:])**2,axis=0)**0.5)
    r_ee = r_ee[np.newaxis]
    jas_grad = (self.b1/(1.+self.b2*r_ee)**2) * (np.outer([1,-1],(r[0,:]-r[1,:])/r_ee).reshape(r.shape))
    return jas_grad

  def local_laplacian(self,r):
    r_ee = (sum((r[0,:]-r[1,:])**2,axis=0)**0.5)
    den = 1+self.b2*r_ee
    jas_val = exp(self.b1*r_ee/den)
    jas_lap = (self.b1/den**2)*(self.b1/den**2+2/r_ee-2*self.b2/den)
    jas_lap = np.array([jas_lap,jas_lap]) #Laplacian is the same for the 2 electrons
    r_ee = r_ee[np.newaxis]
    jas_grad = (self.b1/den**2) * (np.outer([1,-1],(r[0,:]-r[1,:])/r_ee).reshape(r.shape))
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
  test_numeric_analyt(SlaterJastrow(Symm2Orb(1, 1, 1.18, 0.55), Jastrow(0.5, 0.27)))

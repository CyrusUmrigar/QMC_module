import numpy as np

def observables_dict(Z):
    def potential_e(walker):
        coord = walker.coord
        ren=np.sqrt(np.sum(coord**2,axis=1))
        pot_en=np.sum(-Z/ren,axis=0).item(0)
        ree=np.sqrt(np.sum((coord[0,:]-coord[1,:])**2,axis=0))
        pot_ee=1/ree
        return pot_en + pot_ee
    
    def kinetic_e(walker):
        return sum(-0.5 * walker.laplacian)
    
    def total_e(walker):
        return kinetic_e(walker) + potential_e(walker)

    return {"kinetic": kinetic_e, "potential": potential_e, "energy": total_e}

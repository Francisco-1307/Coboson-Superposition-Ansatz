import numpy as np
import scipy as sp

from Auxiliar_Module import Auxiliar

class GaussianModel(Auxiliar):
    """
    Class of models of trapped fermionic dimers with gaussian interaction
    
    """
    
    def __init__(self, g, σ, n_basis_rel, n_basis_Φ, TBM_tensor):
        """Initialize model"""
        self._g           = g
        self._σ           = σ
        self._n_basis_rel = n_basis_rel
        self._n_basis_Φ   = n_basis_Φ
        
        # Calculate H0_rel, V_rel, H_rel and its eigenproblem
        γ = (σ)**(-2.0)         
        H0_rel = np.diag([n + 0.5 for n in range(n_basis_rel)])
        V_rel  = np.zeros((n_basis_rel, n_basis_rel))     
        for m in range(n_basis_rel):
            for n in range(m, n_basis_rel):

                V_rel[m,n] = self.osc_norm_cte_n(m,1.0) * self.osc_norm_cte_n(n,1.0) * self.I_γ(m,n,γ) #mp
#                V_rel[m,n] = self.osc_norm_cte_n(m,1.0) * self.osc_norm_cte_n(n,1.0) * self.I_γ2(m,n,γ)
                V_rel[n,m] = V_rel[m,n]
        
        V_rel = V_rel * (g/(np.sqrt(2*np.pi)*σ))      
        H_rel = H0_rel + V_rel
        
        self._V_rel = V_rel
        self._H_rel = H_rel
        self._eigs  = sp.linalg.eigh(self._H_rel)
        
        self._TBM_tensor = TBM_tensor
        
        self._V_int = np.empty((n_basis_Φ,n_basis_Φ,n_basis_Φ,n_basis_Φ))
        self._V_int_calc = False

    def parameters(self):
        """Print model parameter"""
        print("Model parameters:")
        print(f"--->g           = {self._g}")
        print(f"--->σ           = {self._σ}")
        print(f"--->α_basis     = {1.0}")
        print(f"--->n_basis_rel = {self._n_basis_rel}")
        print(f"--->n_basis_Φ   = {self._n_basis_Φ}")
    
    @property #Getter
    def H_rel(self):
        """don't forget a docstring"""
        return self._H_rel

    @property #Getter
    def eigs(self):
        """don't forget a docstring"""
        return self._eigs
    
    @property #Getter
    def V_rel(self):
        """don't forget a docstring"""
        return self._V_rel
    
    @property #Getter
    def TBM_tensor(self):
        """don't forget a docstring"""
        return self._TBM_tensor   
    
    #Setter
    def V_int(self):
        """Calculate V_nα1,nβ1,nα2,nβ2 tensor"""
        if self._V_int_calc == False:
            n_basis_Φ = self._n_basis_Φ
            # Calculate V_nα1,nβ1,nα2,nβ2 tensor
            TBM_dot_Vrel = np.dot(self.TBM_tensor[:n_basis_Φ,:n_basis_Φ,:,:], self.V_rel)
            TBM_transpose = np.transpose(self.TBM_tensor[:n_basis_Φ,:n_basis_Φ,:,:],axes = [2,3,0,1])
            self._V_int = np.einsum('ijkl,klmn',TBM_dot_Vrel,TBM_transpose)
            self._V_int_calc = True
            return self._V_int
        else:
            return self._V_int
        
    # Instantiate an object of the OneCoboson class
    def oneCoboson(self, n_c, i_ε):
        """ don't forget a docstring """
        return OneCobosonGauss(self._g, self._σ, self._n_basis_Φ, self._eigs, self._TBM_tensor, n_c, i_ε)

    def λ_Scatt (self, n, j, m, i):
        """don't forget a docstring"""
                
        Φ_m = m.WF_CoefMatrix()
        Φ_n = n.WF_CoefMatrix()
        Φ_i = i.WF_CoefMatrix()
        Φ_j = j.WF_CoefMatrix()

        return np.trace(Φ_n.T @ Φ_j @ Φ_m.T @ Φ_i)
    
    def Λ_Scatt (self, n, j, m, i):
        """don't forget a docstring"""
        if i == j:           
            return 2*self.λ_Scatt(n, j, m, i)
        else:
            return self.λ_Scatt(n, j, m, i) + self.λ_Scatt(n, i, m, j)
        
    def ξ_Scatt (self, f_2, i_2, f_1, i_1):
        """don't forget a docstring"""
        
        n_basis_Φ = self._n_basis_Φ

        Φ_f2 = f_2.WF_CoefMatrix()
        Φ_f1 = f_1.WF_CoefMatrix()
        Φ_i2 = i_2.WF_CoefMatrix()
        Φ_i1 = i_1.WF_CoefMatrix()
        
        V_rel = self.V_rel
        
        n_c_f1 = f_1.n_c
        n_c_f2 = f_2.n_c
        rel_eigvec_f1 = f_1.relativeWF_Coefs()
        rel_eigvec_f2 = f_2.relativeWF_Coefs()  
        
        TBMf1 = self.TBM_tensor[:n_basis_Φ,:n_basis_Φ,n_c_f1,:]
        TBMf2 = self.TBM_tensor[:n_basis_Φ,:n_basis_Φ,n_c_f2,:]

        VΦ_f1 = np.dot(TBMf1,V_rel @ rel_eigvec_f1)
        VΦ_f2 = np.dot(TBMf2,V_rel @ rel_eigvec_f2)
        
        matrix_to_trace =  (VΦ_f1.T @ Φ_i1 @ Φ_f2.T @ Φ_i2) + (Φ_f1.T @ Φ_i1 @ VΦ_f2.T @ Φ_i2) \
                         + (VΦ_f1.T @ Φ_i2 @ Φ_f2.T @ Φ_i1) + (Φ_f1.T @ Φ_i2 @ VΦ_f2.T @ Φ_i1) 
            
        return -0.5*np.trace(matrix_to_trace)  
        
    def ξIn_Scatt (self, f_2, i_2, f_1, i_1):
        """don't forget a docstring"""
        
        Φ_f2 = f_2.WF_CoefMatrix()
        Φ_f1 = f_1.WF_CoefMatrix()
        Φ_i2 = i_2.WF_CoefMatrix()
        Φ_i1 = i_1.WF_CoefMatrix()
        
        # V_ijkl
        V   = self.V_int()
        # V_ikjl
        V_T = np.transpose(V, axes = [0,2,1,3])
        
        ξIn_aux =  np.einsum('ij,ij', (Φ_f1 @ Φ_i1.T), np.einsum('ijkl,kl', V_T, (Φ_f2.T @ Φ_i2)) ) \
                 + np.einsum('ij,ij', (Φ_f1 @ Φ_i2.T), np.einsum('ijkl,kl', V_T, (Φ_f2.T @ Φ_i1)) ) \
                 + np.einsum('ij,ij', (Φ_f2 @ Φ_i2.T), np.einsum('ijkl,kl', V_T, (Φ_f1.T @ Φ_i1)) ) \
                 + np.einsum('ij,ij', (Φ_f2 @ Φ_i1.T), np.einsum('ijkl,kl', V_T, (Φ_f1.T @ Φ_i2)) )
        
        return -0.5 * ξIn_aux
    
    # Here, I do NOT normalize the two-coboson states!!! 
    # ρ^(1) = <f_1,f_2|n_{*,*}|i_1,i_2>
    def TwoCobosonSpace_FermionOccupationMatrix (self, f_2, i_2, f_1, i_1):
        """don't forget a docstring"""

        Φ_f2 = f_2.WF_CoefMatrix()
        Φ_f1 = f_1.WF_CoefMatrix()
        Φ_i2 = i_2.WF_CoefMatrix()
        Φ_i1 = i_1.WF_CoefMatrix()

        ρ_1 = (Φ_f1 @ Φ_i1.T)*self.Kronecker_δ(f_2,i_2) + self.Kronecker_δ(f_1,i_2)*(Φ_f2 @ Φ_i1.T)\
             +self.Kronecker_δ(f_1,i_1)*(Φ_f2 @ Φ_i2.T) + (Φ_f1 @ Φ_i2.T)*self.Kronecker_δ(f_2,i_1)

        ρ_1 = ρ_1 - (Φ_f1 @ Φ_i2.T @ Φ_f2 @ Φ_i1.T) - (Φ_f2 @ Φ_i2.T @ Φ_f1 @ Φ_i1.T) \
                  - (Φ_f2 @ Φ_i1.T @ Φ_f1 @ Φ_i2.T) - (Φ_f1 @ Φ_i1.T @ Φ_f2 @ Φ_i2.T)
        
        return ρ_1

    
class OneCobosonGauss(Auxiliar):
    """don't forget a docstring"""
    def __init__(self, g, σ, n_basis_Φ, eigs, TBM_tensor, n_c, i_ε):

        self._g          = g
        self._σ          = σ
        self._n_basis_Φ  = n_basis_Φ
        self._eigs       = eigs
        self._n_c        = n_c
        self._i_ε        = i_ε
        self._TBM_tensor = TBM_tensor

        self._Φ_Calc   = False
        self._Φ_Matrix = np.empty((n_basis_Φ,n_basis_Φ))

    @property #Getter
    def n_c(self):
        return self._n_c

    @property #Getter
    def i_ε(self):
        return self._i_ε
    
    @property #Getter
    def n_basis_Φ(self):
        return self._n_basis_Φ

    def __eq__(self, other):
        return (self._g == other._g and 
                self._σ == other._σ and 
                self._n_basis_Φ == other._n_basis_Φ and 
                self._n_c == other._n_c and 
                self._i_ε == other._i_ε)            
    
    def relativeEnergy(self):
        """don't forget a docstring"""
        return self._eigs[0][self._i_ε]

    def relativeWF_Coefs(self):
        """don't forget a docstring"""
        return self._eigs[1][:,self._i_ε]

    def relativeWF(self, zr):
        """don't forget a docstring"""
        rel_eigvec = self.relativeWF_Coefs()
        phi_alpha_n_arr = np.array([self.phi_alpha_n (n,1.0,zr) for n in range(rel_eigvec.size)])

        return np.dot(rel_eigvec,phi_alpha_n_arr)        

    def energy(self):
        """don't forget a docstring"""
        return (self.n_c + 0.5) + self._eigs[0][self._i_ε]
    
    def WF_CoefMatrix(self):
        """ Method for constructing an 2d-array of the coefficients <nα,nβ|Φ>"""
        if self._Φ_Calc == False:
            n_c = self.n_c
            rel_eigvec = self.relativeWF_Coefs()
            for nα in range(self.n_basis_Φ):
                for nβ in range(self.n_basis_Φ): 
                    n_r_aux = nα+nβ-n_c
                    if n_r_aux >= 0 and n_r_aux < rel_eigvec.size:
                        self._Φ_Matrix[nα,nβ] = self._TBM_tensor[nα,nβ,n_c,n_r_aux]*rel_eigvec[n_r_aux]
                    else:
                        self._Φ_Matrix[nα,nβ] = 0.
            self._Φ_Calc = True            
            return self._Φ_Matrix
        else:
            return self._Φ_Matrix
        
    def WF(self, z1, z2):
        """don't forget a docstring"""
        z_c = (z1+z2)/np.sqrt(2.0)
        z_r = (z1-z2)/np.sqrt(2.0)

        return self.phi_alpha_n (self._n_c,1.0,z_c)*self.relativeWF(z_r)

    def OneBodyRDM(self):
        """Calculates the One-body Reduced Density Matrix (RDM) in the ϕ_{n_α} basis"""

        Φ = self.WF_CoefMatrix()
        ρ_1 = np.dot(Φ,Φ.T)

        return ρ_1

    def OneBodyDensity(self, ρ_1, z_arr):
        """don't forget a docstring"""

        ρ_1 = self.OneBodyRDM()
        OneBodyDensity = []        
        for z in z_arr:            
            phi_n_arr = np.array([self.phi_alpha_n (n,1.0,z) for n in range(self.n_basis_Φ)])
            OneBodyDensity.append(np.dot(phi_n_arr, np.dot(ρ_1, phi_n_arr)))

        return np.array(OneBodyDensity)       

    def OneBodyRDM_EigenDecomp(self):
        """don't forget a docstring"""

        ρ_1 = self.OneBodyRDM()

        # Diagonalizes ρ_1 = ΦΦ^T: 
        #----->> eigenvalues: λ_i -> Schmidt values 
        #----->> eigenvector: (C_0,C_1,...,C_n) -> expansion coeff of natural orbitals in ϕ_k in the oscilator basis

        schm_val_orb_aux = sp.linalg.eigh(ρ_1, eigvals_only = False)

        # sequence of λ_i
        schm_val  = schm_val_orb_aux[0]
        # 2d array with coeffs (C_0,C_1,...,C_n) in the collumns
        schm_orb_coefs = schm_val_orb_aux[1]

        # Dictionary of Schm values λ_i's and its relatated natural orbitals coeffs (C_0,C_1,...,C_n)
        schm_dic = {key:value for key,value in zip(schm_val,schm_orb_coefs.T)}

        # dictionary sorted by keys (λ_i's)
        schm_dic = dict(sorted(schm_dic.items(), reverse = True))

        return schm_dic
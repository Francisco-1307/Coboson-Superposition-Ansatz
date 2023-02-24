import numpy as np
import math
import scipy as sp
from scipy.special import eval_hermite, factorial, binom
import mpmath as mp # Package for multiple precision

class Auxiliar():
    
    @staticmethod
    def cm_in_inches():
        return 1/2.54  # centimeters in inches

    @staticmethod
    def truncate(number, digits):
        stepper = 10.0 ** digits
        return math.trunc(stepper * number) / stepper

    @staticmethod
    def mp_to_float (x): 
        return float(mp.nstr(x,30))

    @staticmethod
    def bigamma (x): 
        return sp.special.psi(x)

    @staticmethod
    def Parabolic_Cylinder (n,x):
        return sp.special.pbdv(n,x)[0]

    @staticmethod
    def eval_hermit_norm (n,x):
        """n-th normalized Hermite polynomial"""
        norm_cte = ((np.pi**0.5)*(2**n)*factorial(n))**(-0.5)
        return norm_cte*eval_hermite(n,x)

    @staticmethod
    def osc_norm_cte_n (n,α):
        """Normalization constant"""
        return np.sqrt(α)*((np.pi**0.5)*(2**n)*factorial(n))**(-0.5)

    @staticmethod
    def phi_alpha_n (n,α,x):
        """α-deformed n-th harmonic oscillator function"""
        norm_cte = np.sqrt(α)*((np.pi**0.5)*(2**n)*factorial(n))**(-0.5)
        return norm_cte*eval_hermite(n,α*x)*np.exp(-0.5*(α*x)**2)
 
    #Limit g = -\infty: Lieb–Liniger gas of hard-core bosonic dimers
    @staticmethod
    def onebody_dens_LL(N,x):

        phi_arr = np.array([Auxiliar.phi_alpha_n(j,np.sqrt(2.0), x) for j in range(N)])
        phi_arr = np.absolute(phi_arr)**2
        
        return 2*phi_arr.sum()/2

    #Non-interacting 
    @staticmethod   
    def onebody_dens_0(N,x):
        phi_arr = np.array([Auxiliar.phi_alpha_n(j,1.0, x) for j in range(N)])
        phi_arr = np.absolute(phi_arr)**2
        
        return 2*phi_arr.sum()/2

    #Limit g = +\infty: Tonks-Girardeau gas of ‘fermionized' dimers
    @staticmethod
    def onebody_dens_TG(N,x):
        phi_arr = np.array([Auxiliar.phi_alpha_n(j,1.0, x) for j in range(2*N)])
        phi_arr = np.absolute(phi_arr)**2
        
        return phi_arr.sum()/2

    @staticmethod
    def OneBodyDensity_from_ρ_1 (ρ_1, z_arr):
        """don't forget a docstring"""
        
        n_basis = ρ_1.shape[0]
        
        OneBodyDensity = []        
        for z in z_arr:            
            phi_n_arr = np.array([Auxiliar.phi_alpha_n (n,1.0,z) for n in range(n_basis)])
            OneBodyDensity.append(np.dot(phi_n_arr, np.dot(ρ_1, phi_n_arr)))

        return np.array(OneBodyDensity)  

    @staticmethod
    def Kronecker_δ (i,j):
        
        return 1. if (i==j) else 0.

    @staticmethod
    def I_γ(n, m, γ):
        """
            Attention: Uses the mp.math package!
        """
        γ_mp = mp.mpf(γ)

        if   ( (n % 2 == 0) and (m % 2 == 0) ):

            summand = lambda l: 2.0**l*(-γ_mp)**((n+m)/2-l)/(mp.factorial(l)*mp.factorial((n-l)/2)*mp.factorial((m-l)/2)) if l % 2 == 0 else mp.mpf(0.0)
            I = mp.sqrt(mp.pi)*mp.factorial(n)*mp.factorial(m)*(1.0+γ_mp)**(-(n+m+1)/2)*mp.nsum(summand, [0,min(n,m)])

            return float(mp.nstr(I,30))

        elif ( (n % 2 == 1) and (m % 2 == 1) ):

            summand = lambda l: 2.0**l*(-γ_mp)**((n+m)/2-l)/(mp.factorial(l)*mp.factorial((n-l)/2)*mp.factorial((m-l)/2)) if l % 2 == 1 else mp.mpf(0.0)
            I = mp.sqrt(mp.pi)*mp.factorial(n)*mp.factorial(m)*(1.0+γ_mp)**(-(n+m+1)/2)*mp.nsum(summand, [0,min(n,m)])

            return float(mp.nstr(I,30))

        else:

            return 0.0        

    @staticmethod        
    def I_γ2(m, n, γ):
        """don't forget a docstring"""
        
        if (m+n) % 2 == 0:
            
            summand = []
            for k in range(0, math.floor(n/2)+1):
                for k1 in range(0, math.floor(m/2)+1):                   
                    summand.append( (factorial(k)*factorial(k1)*factorial(n-2*k)*factorial(m-2*k1))**(-1)
                                    *eval_hermite(m+n-2*k-2*k1,0.) * (1+γ)**(k+k1) )
                    
            summand = np.array(summand)           
            return (-1)**(0.5*(m+n))*np.sqrt(np.pi)*factorial(m)*factorial(n)*(1+γ)**(-0.5*(m+n+1)) * summand.sum()     
        else:
            return 0.
    
    @staticmethod
    def TBM (n1, n2, N, n):
        """
            Talmi-Brodi-Moshinsky coefficients: 
            TBM(n1, n2, N, n) = <n_1,n_2|n_c,n_r>
        """
        if (n1+n2 == N+n):

            summad = np.array([(-1)**m * binom(N,n1-n+m) * binom(n,m) for m in range(min(n,n2)+1)])
            return np.sqrt(factorial(n1)*factorial(n2)/(2**(n1+n2)*factorial(N)*factorial(n))) * summad.sum() 

        else:

            return 0.

    @staticmethod
    def TBM_mp (n1, n2, N, n):
        """
            # Attention: Uses the mp.math package!
            Talmi-Brodi-Moshinsky coefficients: 
            TBM(n1, n2, n_c, n_r) = <n_1,n_2|n_c,n_r>
        """
        if (n1+n2 == N+n):

            summad = lambda m: (-1)**m * mp.binomial(N,n1-n+m) * mp.binomial(n,m)
            summation = mp.sqrt(mp.factorial(n1)*mp.factorial(n2)/(2**(n1+n2)*mp.factorial(N)*mp.factorial(n)))*mp.nsum(summad,[0,min(n,n2)])
            return float(mp.nstr(summation,30))

        else:

            return 0.0

    @staticmethod       
    def TBM_tensor_calc(n_basis_c, n_basis_rel):
        """
            Calculate Talmi-Brodi-Moshinsky tensor
            TBM(n_α,n_β,n_c,n_r) = <n_α,n_β|n_c,n_r>
        """
        # Attention: n_c ranges from 0 to n_basis_c   -1 --> range(n_basis_c)
        #            n_r ranges from 0 to n_basis_rel -1 --> range(n_basis_rel)
        #            n_α,n_β ranges from 0 to (n_basis_c -1) + (n_basis_rel -1) --> range(n_basis_c + n_basis_rel - 1)
        n_basis_Φ = n_basis_c + n_basis_rel - 1
        TBM_tensor = np.zeros((n_basis_Φ,n_basis_Φ,n_basis_c,n_basis_rel))
        for n_c in range(n_basis_c):
            for n_r in range(n_basis_rel):
                # Attention: n_α,n_β ranges from 0 to n_c + n_r
                for n_α in range(n_c + n_r + 1):
                    for n_β in range(n_c + n_r + 1):
                        TBM_tensor[n_α,n_β,n_c,n_r] = Auxiliar.TBM(n_α,n_β,n_c,n_r) 
        return TBM_tensor

    @staticmethod       
    def TBM_tensor_calc_mp(n_basis_c, n_basis_rel):
        """
            Attention: Uses the mp.math package! 
            Calculate Talmi-Brodi-Moshinsky tensor
            TBM(n_α,n_β,n_c,n_r) = <n_α,n_β|n_c,n_r> 
        """
        # Attention: n_c ranges from 0 to n_basis_c   -1 --> range(n_basis_c)
        #            n_r ranges from 0 to n_basis_rel -1 --> range(n_basis_rel)
        #            n_α,n_β ranges from 0 to (n_basis_c -1) + (n_basis_rel -1) --> range(n_basis_c + n_basis_rel - 1)
        n_basis_Φ = n_basis_c + n_basis_rel - 1
        TBM_tensor = np.zeros((n_basis_Φ,n_basis_Φ,n_basis_c,n_basis_rel))
        for n_c in range(n_basis_c):
            for n_r in range(n_basis_rel):
                # Attention: n_α,n_β ranges from 0 to n_c + n_r
                for n_α in range(n_c + n_r + 1):
                    for n_β in range(n_c + n_r + 1):
                        TBM_tensor[n_α,n_β,n_c,n_r] = Auxiliar.TBM_mp(n_α,n_β,n_c,n_r) 
        return TBM_tensor


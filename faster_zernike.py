# Zernike radial direct computes the Zernike radial polynomials and respective derivatives
def zernike_radial_direct(r, n, m, dr):
    """Radial part of zernike polynomials. Modified Prata's algorithm."""
    
    m = abs(m) # m must be positive

    # Define a 3D matrix to save the radial polynomials R_n^m(rho)
    # and respective deritivatives d^l( R_n^m(rho) )/drho^l
    # r indicates the value of R_n^m(rho = r)
    R = np.zeros((len(n), len(m), len(r)))  # R_nmk

    # Generate respective matrices if derivatives of R are to be computed
    if dr > 0:
        d1R = np.zeros((len(n), len(m), len(r)))  # d(R_nmk)/dr
    elif dr > 1:
        d2R = np.zeros((len(n), len(m), len(r)))  # d2(R_nmk)/dr2
    elif dr > 2:
        d3R = np.zeros((len(n), len(m), len(r)))  # d3(R_nmk)/dr3
    elif dr > 3:
        d4R = np.zeros((len(n), len(m), len(r)))  # d4(R_nmk)/dr4

    # First step: Compute R_n^n using the simple relation R[n,n] = rho^n
    for n_i in (0, np.max(n)+1):
        
        # Compute 0th-derivative
        if n_i == 0:
            R[n_i, n_i] = np.ones(r.size)
        else:
            R[n_i, n_i] = r ** n_i       
        # Compute 1st-derivative
        if dr > 0:
            if n_i == 0:
                d1R[n_i, n_i] = np.zeros(r.size)
            elif n_i == 1:
                d1R[n_i, n_i] = n_i * np.ones(r.size)
            else:
                d1R[n_i, n_i] = n_i * r ** (n_i - 1)       
        # Compute 2nd-derivative
        elif dr > 1:
            if n_i <= 1:
                d2R[n_i, n_i] = np.zeros(r.size)
            elif n_i == 2:
                d2R[n_i, n_i] = n_i*(n_i - 1) * np.ones(r.size)
            else:
                d2R[n_i, n_i] = n_i*(n_i - 1) * r ** (n_i - 2)        
        # Compute 3rd-derivative
        elif dr > 2:
            if n_i <= 2:
                d3R[n_i, n_i] = np.zeros(r.size)
            elif n_i == 3:
                d3R[n_i, n_i] = n_i*(n_i - 1)*(n_i - 2) * np.ones(r.size)
            else:
                d3R[n_i, n_i] = n_i*(n_i - 1) * r ** (n_i - 2)
        # Compute 4th-derivative
        elif dr > 3:
            if n_i <= 3:
                d4R[n_i, n_i] = np.zeros(r.size)
            elif n_i == 4:
                d4R[n_i, n_i] = n_i*(n_i - 1)*(n_i - 2)*(n_i - 3) * np.ones(r.size)
            else:
                d4R[n_i, n_i] = n_i*(n_i - 1)*(n_i - 2)*(n_i - 3) * r ** (n_i - 4)
    
    # Second step: Compute R[n,0] using the conventional relation for R_n^m(rho)
    for n_i in (1, np.max(n) + 1):
        
        # Right now we are using the non-optimized zernike radial evaluation
        # When it's ready, we should switch to the optimized version
        
        R[n_i, 0] = zernike_radial(r, n_i, 0, 0)
    
        # Compute 1st-derivative
        if dr > 0:
            d1R[n_i, 0] = zernike_radial(r, n_i, 0, 1)
        # Compute 2nd-derivative
        elif dr > 1:
            d2R[n_i, 0] = zernike_radial(r, n_i, 0, 2)
        # Compute 3rd-derivative
        elif dr > 2:
            d3R[n_i, 0] = zernike_radial(r, n_i, 0, 3)       
        # Compute 4th-derivative
        elif dr > 3:
            d4R[n_i, 0] = zernike_radial(r, n_i, 0, 4)
    
    # Third step: Compute the remaining R_n^m
    for n_i in (3, np.max(n) + 1):

        for m_i in (1, np.max(m)):

            # There is a condition that dictates when to use the recursive or the direct relation
            if  n_i < m_i + 2:
                
                R[n_i, m_i] = zernike_radial(r, n_i, m_i, 0)
                
                # Compute 1st-derivative
                if dr > 0:
                    d1R[n_i, m_i] = zernike_radial(r, n_i, m_i, 1)
                # Compute 2nd-derivative
                elif dr > 1:
                    d2R[n_i, m_i] = zernike_radial(r, n_i, m_i, 2)
                # Compute 3rd-derivative
                elif dr > 2:
                    d3R[n_i, m_i] = zernike_radial(r, n_i, m_i, 3)               
                # Compute 4th-derivative
                elif dr > 3:
                    d4R[n_i, m_i] = zernike_radial(r, n_i, m_i, 4)
            
            elif m_i + 2 <= n_i < m_i:

                K_1 = 2 * n_i / (m_i + n_i)
                K_2 = 1 - K_1
                
                # Compute 0th-derivative
                R[n_i, m_i] = r * K_1 * R[n_i - 1, m_i - 1] + K_2 * R[n_i - 2, m_i]
            
                # Compute 1st-derivative
                if dr > 0:
                    # Recursive relation
                    d1R[n_i, m_i] = (K_1 * (R[n_i - 1, m_i - 1] + r * d1R[n_i - 1, m_i - 1])
                                     + K_2 * d1R[n_i - 2, m_i])
                # Compute 2nd-derivative
                elif dr > 1:
                    # Recursive relation
                    d2R[n_i, m_i] = (K_1 * ((dr-1)*d1R[n_i - 1, m_i - 1] 
                                            + r * d2R[n_i - 1, m_i - 1])
                                     + K_2 * d2R[n_i - 2, m_i])
                # Compute 3rd-derivative
                elif dr > 2:
                    # Recursive relation
                    d3R[n_i, m_i] = (K_1 * ((dr-1)*d2R[n_i - 1, m_i - 1] 
                                            + r * d3R[n_i - 1, m_i - 1])
                                     + K_2 * d3R[n_i - 2, m_i])
                # Compute 4th-derivative
                elif dr > 3:
                    # Recursive relation
                    d4R[n_i, m_i] = (K_1 * ((dr-1)*d3R[n_i - 1, m_i - 1] 
                                            + r * d4R[n_i - 1, m_i - 1])
                                     + K_2 * d4R[n_i - 2, m_i])
    
        # Define the output depending on the derivative desired
        if dr == 0:
            out = R
        elif dr == 1:
            out = d1R
        elif dr == 2:
            out = d2R
        elif dr == 3:
            out = d3R
        elif dr == 4:
            out = d4R

    return out
# Zernike radial direct computes the Zernike radial polynomials and respective derivatives
def zernike_radial_direct(r, n, m, dr):
    """Radial part of zernike polynomials. Modified Prata's algorithm."""
    
    m = abs(m) # m must be positive

    # Define a 3D matrix to save the radial polynomials R_n^m(rho)
    # and respective deritivatives d^l( R_n^m(rho) )/drho^l
    # r indicates the value of R_n^m(rho = r)
    R = np.zeros((len(n), len(m), len(r)))  # R_nmk

    # Generate a matrix if the first derivatives of R are to be computed
    if dr == 1:
        d1R = np.zeros((len(n), len(m), len(r)))  # dR_nmk

    # First step: Compute R_n^n using the simple relation R[n,n] = rho^n
    for n_i in (0, np.max(n)+1):
        
        if n_i == 0:
            R[n_i, n_i] = np.ones(r.size)
        else:
            R[n_i, n_i] = r ** n_i

        if dr == 1:
            if n_i == 0:
                d1R[n_i, n_i] = np.zeros(r.size)
            elif n_i == 1:
                d1R[n_i, n_i] = n_i * np.ones(r.size)
            else:
                d1R[n_i, n_i] = n_i * r ** (n_i - 1)
    
    # Second step: Compute R[n,0] using the conventional relation for R_n^m(rho)
    for n_i in (1, np.max(n) + 1):
        
        # Right now we are using the non-optimized zernike radial evaluation
        # When it's ready, we should switch to the optimized version
        
        R[n_i, 0] = zernike_radial(r, n_i, 0, 0)
    
        if dr == 1:
            d1R[n_i, 0] = zernike_radial(r, n_i, 0, dr)
    
    # Third step: Compute the remaining R_n^m
    for n_i in (3, np.max(n) + 1):

        for m_i in (1, np.max(m)):

            # There is a condition that dictates when to use the recursive or the direct relation
            if  n_i < m_i + 2:
                
                R[n_i, 0] = zernike_radial(r, n_i, m_i, dr)
                #out = R
            
            elif m_i + 2 <= n_i < m_i:

                K_1 = 2 * n_i / (m_i + n_i)
                K_2 = 1 - K_1
                
                R[n_i, m_i] = r * K_1 * R[n_i - 1, m_i - 1] + K_2 * R[n_i - 2, m_i]
            
                if dr == 1:
                    # Recursive relation
                    d1R[n_i, m_i] = (K_1 * R[n_i - 1, m_i - 1]
                                     + r * K_1 * d1R[n_i - 1, m_i - 1]
                                     + K_2 * d1R[n_i - 2, m_i])
    
        # Define the output depending on the derivative desired
        if dr == 0:
            out = R
        elif dr == 1:
            out = d1R

    return out
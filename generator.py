#### 2-D random field generator
import numpy as np

def generator(nx, dx, lx, ang, sigma2, mu):
    
    dimen   = 2
    ntot    = np.prod(nx[0:dimen])
    
    # ============== BEGIN COVARIANCE BLOCK ==========================================

    ang2=ang/180*np.pi
    # Grid in Physical Coordinates
    y = np.arange(-nx[0]/2*dx[0], (nx[0]-1)/2*dx[0], dx[0])
    x = np.arange(-nx[1]/2*dx[1], (nx[1]-1)/2*dx[1], dx[1])

    X, Y = np.meshgrid(x, y)

    # Transformation to angular coordinates
    X2 = np.cos(ang2)*X + np.sin(ang2)*Y;
    Y2 =-np.sin(ang2)*X + np.cos(ang2)*Y;

    # Accounting for corellation lengths
    H = np.sqrt((X2/lx[0])**2+(Y2/lx[1])**2);

    # Matérn 3/2
    RYY = sigma2 * np.multiply((1+np.sqrt(3)*H),np.exp(-np.sqrt(3)*H))

    # ============== END COVARIANCE BLOCK =====================================
    
    # ============== BEGIN POWER-SPECTRUM BLOCK ===============================
    # Fourier Transform (Origin Shifted to Node (0,0))
    # Yields Power Spectrum of the field
    SYY=np.fft.fftn(np.fft.fftshift(RYY))/ntot;
    # Remove Imaginary Artifacts
    SYY=np.abs(SYY)
    SYY[0,0] =0;
    # ============== END POWER-SPECTRUM BLOCK =================================
       
    # ============== BEGIN FIELD GENERATION BLOCK =============================
    # Generate the field
    # nxhelp is nx with the first two entries switched
    nxhelp = nx[0:dimen].T;

    if dimen > 1:
        nxhelp[0:2] = [nx[0], nx[1]]
    else:
        nxhelp = np.array([1,nx[0]]).T;

    # Generate the real field according to Dietrich and Newsam (1993)
    ran = np.multiply(np.sqrt(SYY), np.squeeze(
            np.array([np.random.randn(nxhelp[0], nxhelp[1]) + 
                      1j*np.random.randn(nxhelp[0], nxhelp[1])] 
                     ,dtype = 'complex_'))
            )
    # Backtransformation into the physical coordinates
    ran = np.real(np.fft.ifftn(ran*ntot))+mu;
    # ============== END FIELD GENERATION BLOCK ===============================
    
    return ran
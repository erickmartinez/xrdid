import numpy as np
import xrdid.errors as errors

# A compound type to store matrix data containing the diffractogram's 2theta,intensity,Q-values as columns
xrd_type = np.dtype([('2theta', 'd'), ('intensity', 'd'), ('Q', 'd')])
# A compound type to store the generated R values
qr_type = np.dtype([('Qi', 'd'), ('Qi_id', 'i'), ('m', 'i'), ('n', 'i'), ('R', 'd'), ('phi', 'd')])


class PeakId:
    """
    This class defines basic peak identification functionality

    Attributes
    ----------
    _2theta: np.ndarray
        The 2theta positions of the peaks from the diffractogram (degrees)
    _intensity: np.ndarray
        The values of the intensities of the peaks from the diffractogram
    lam: float
        The wavelength of the XRD source (Å)
    _aStar: float
        The reciprocal space lattice constant 1/a
    _bStar: float
        The reciprocal space lattice constant 1/b
    _cStar: float
        The reciprocal space lattice constant 1/c
    _cosAlphaStar: float
        The cosine of reciprocal alpha angle of the unit cell (degrees)
    _cosBetaStar: float
        The cosine of reciprocal beta angle of the unit cell (degrees)
    _cosGammaStar: float
        The cosine of reciprocal beta angle of the unit cell (degrees)
    npoints: int
        The number of experimental data points

    Methods
    -------
    q_theta:
        Estimates Q(theta) = 4 sin^2(theta)/lambda^2
    q_hkl:
        Estimates Q(hkl) = h^2 * a*^2 + k^2 * b*^2 + l^2 * c*^2 + 2*k*l*(b*)*(c*)*cos
    """
    _2theta: np.ndarray = None
    _intensity: np.ndarray = None
    lam: float = 1.540593
    _aStar: float = None
    _bStar: float = None
    _cStar: float = None
    _cosAlphaStar: float = None
    _cosBetaStar: float = None
    _cosGammaStar: float = None
    _cosAlpha: float = None
    _cosBeta: float = None
    _cosGamma: float = None
    _npoints: int = 0
    _mmax: int = 4
    _nmax: int = 4

    def __init__(self, two_theta: np.ndarray, intensity: np.ndarray, lam: float = 1.540593):
        """
        Parameters
        ----------
        two_theta: np.ndarray
            The 2theta column in degrees
        intensity:
            The intensity column of the diffractogram
        lam:
            The wavelength of the X-ray source (in Å)
        """
        self._2theta = two_theta
        self._intensity = intensity
        self.lam = lam
        self._npoints = len(two_theta)
        xrd_data = np.empty(self._npoints, dtype=xrd_type)
        for i, x, y in zip(range(self._npoints), two_theta, intensity):
            xrd_data[i] = (x, y, self.q_theta(x/2.0))
        xrd_data.sort(order='Q')
        self._xrd_data = xrd_data

    def q_theta(self, theta: float) -> float:
        """
        This method returns Q = Q(theta) as a function of the diffraction angle

        Parameters
        ----------
        theta: float
            The diffraction angle

        Returns
        -------
        float:
            The value of Q
        """
        return 4.0*np.power(np.sin(np.pi*theta/180)/self.lam, 2.0)*1E4

    def q_hkl(self, h: int, k: int, l: int) -> float:
        """
        This method returns Q(hkl) provided that we have valid values for the reciprocal lattice vectors  a*, b*, c* and
        the angles alpha, beta and gamma.

        Parameters
        ----------
        h: int
            The Miller index h
        k: int
            The Miller index k
        l: int
            The Miller index l

        Returns
        -------
        float:
            The estimated Q(hkl)

        Raises
        ------
        errors.InvalidClassProperty
            If either a*, b*, c*, alpha, beta or gamma are undefined.
        """
        if self._aStar is None:
            raise errors.InvalidClassProperty(property='a*', value=self._aStar)
        if self._bStar is None:
            raise errors.InvalidClassProperty(property='b*', value=self._bStar)
        if self._cStar is None:
            raise errors.InvalidClassProperty(property='c*', value=self._cStar)
        if self._cosAlphaStar is None:
            raise errors.InvalidClassProperty(property='cos(alpha*)', value=self._cosAlphaStar)
        if self._cosBetaStar is None:
            raise errors.InvalidClassProperty(property='cos(beta*)', value=self._cosBetaStar)
        if self._cosGammaStar is None:
            raise errors.InvalidClassProperty(property='cos(gamma*)', value=self._cosGammaStar)

        q = np.power(h*self._aStar, 2.0) + np.power(k*self._bStar, 2.0) + np.power(l*self._bStar, 2.0)
        q += 2.0 * k * l * self._bStar * self._cStar * self._cosAlphaStar
        q += 2.0 * l * h * self._cStar * self._aStar * self._cosBetaStar
        q += 2.0 * h * k * self._aStar * self._bStar * self._cosGammaStar

        return q*1E4

    def estimate_r(self, idx_q1: int, idx_q2: int, idx_qi: int, m: int, n: int) -> float:
        """
        Estimates R for a Qi, given Q', Q'' and the values m and n.

        Parameters
        ----------
        idx_q1: int
            The index of the data line to get Q' from
        idx_q2: int
            The index of the data line to get Q'' from
        idx_qi: int
            The index of the data line to get Qi from
        m: int
            The value of m
        n: int
            The value of n

        Returns
        -------
        float:
            R

        Raises
        ------
        ValueError:
            If |idx_q1| > number of points
        ValueError:
            If |idx_q2| > number of points
        ValueError:
            If |idx_qi| > number of points
        ValueError:
            If idx_qi == idx_q1 or idx_qi = idx_q2
        """
        # First check that the indices are valid
        if abs(idx_q1) + 1 > self._npoints:
            raise ValueError('The first index: \'{0:d}\' is out of range (max: {1:d}).'.format(idx_q1, self._npoints))
        elif abs(idx_q2) + 1 > self._npoints:
            raise ValueError('The second index: \'{0:d}\' is out of range (max: {1:d}).'.format(idx_q2, self._npoints))
        elif abs(idx_qi) + 1 > self._npoints:
            raise ValueError('The second index: \'{0:d}\' is out of range (max: {1:d}).'.format(idx_q2, self._npoints))
        if idx_q1 == idx_qi or idx_qi == idx_q2:
            raise ValueError('The value of Qi should be different than Q\', and Q\'\'.')

        q1 = self._xrd_data[idx_q1]['Q']
        q2 = self._xrd_data[idx_q2]['Q']
        qi = self._xrd_data[idx_qi]['Q']

        return np.abs((np.power(m, 2.0)*q1 + np.power(n, 2.0)*q2 - qi)/(2.0*m*n))

    def r_table(self, idx_q1: int, idx_q2: int):
        # First check that the indices are valid
        if abs(idx_q1) + 1 > self._npoints:
            raise ValueError('The first index: \'{0:d}\' is out of range (max: {1:d}).'.format(idx_q1, self._npoints))
        elif abs(idx_q2) + 1 > self._npoints:
            raise ValueError('The second index: \'{0:d}\' is out of range (max: {1:d}).'.format(idx_q2, self._npoints))

        n_values = [i for i in range(-self._nmax, self._nmax + 1) if i != 0]
        n_rows = (self._npoints - 2) * self._mmax * len(n_values)
        r_table = np.empty(n_rows, dtype=qr_type)
        j = 0
        for i in range(self._npoints):
            if i != idx_q1 and i != idx_q2:
                for m in range(1, self._mmax + 1):
                    for n in n_values:
                        r = self.estimate_r(idx_q1=idx_q1, idx_q2=idx_q2, idx_qi=i, m=m, n=n)
                        r_table[j] = (self._xrd_data[i]['Q'], i, m, n, np.round(r, 1),
                                      np.round(self.phi_r12(idx_q1, idx_q2, r), 2))
                        j += 1

        # Delete invalid rows
        idx_valid = r_table['phi'] > 0.0
        r_table = r_table[idx_valid]

        return r_table

    def phi_r12(self, idx_q1: int, idx_q2: int, r: float) -> float:
        # First check that the indices are valid
        if abs(idx_q1) + 1 > self._npoints:
            raise ValueError('The first index: \'{0:d}\' is out of range (max: {1:d}).'.format(idx_q1, self._npoints))
        elif abs(idx_q2) + 1 > self._npoints:
            raise ValueError('The second index: \'{0:d}\' is out of range (max: {1:d}).'.format(idx_q2, self._npoints))

        q1 = self._xrd_data[idx_q1]['Q']
        q2 = self._xrd_data[idx_q2]['Q']

        cos_phi = r / np.sqrt(q1*q2)
        # print('Q\' = {0}, Q\'\' = {1}, r = {2}, cos(phi) = {3}, phi = {4}'.format(q1, q2, r, cos_phi,
        #                                                                           np.arccos(cos_phi)*180.0/np.pi % 180))
        if np.abs(cos_phi) > 1.0:
            return -1
        return np.arccos(cos_phi)*180.0/np.pi % 180

    def estimate_a_star(self, q100: float) -> float:
        """
        Estimate the value of a*, provided a known value of Q(100)

        Parameters
        ----------
        q100: float
            The value Q(100) = h^2 * a*^2

        Returns
        -------
        float:
            a*
        """
        a_star = np.sqrt(q100)
        self._aStar = a_star
        return a_star

    def estimate_b_start(self, q010: float) -> float:
        """
        Estimate the value of b*, provided a known value of Q(010)

        Parameters
        ----------
        q010: float
            The value Q(010) = k^2 * b*^2

        Returns
        -------
        float:
            b*
        """
        b_star = np.sqrt(q010)
        self._bStar = b_star
        return b_star

    def estimate_c_start(self, q001: float) -> float:
        """
        Estimate the value of c*, provided a known value of Q(001)

        Parameters
        ----------
        q001: float
            The value Q(001) = l^2 * c*^2

        Returns
        -------
        float:
            c*
        """
        c_star = np.sqrt(q001)
        self._cStar = c_star
        return c_star

    @property
    def cos_alpha(self) -> float:
        if self._cosAlpha is None:
            if self.validate_reciprocal_angles():
                cos_alpha = (self._cosBetaStar * self._cosGammaStar - self._cosAlphaStar)
                cos_alpha /= (self._cosBetaStar * self._cosGammaStar)
                self._cosAlpha = cos_alpha
                return cos_alpha
        else:
            return self._cosAlpha

    @property
    def cos_beta(self) -> float:
        if self._cosBeta is None:
            if self.validate_reciprocal_angles():
                cos_beta = (self._cosGammaStar * self._cosAlphaStar - self._cosBetaStar)
                cos_beta /= (self._cosGammaStar * self._cosAlphaStar)
                self._cosBeta = cos_beta
                return cos_beta
        else:
            return self._cosBeta

    @property
    def cos_gamma(self) -> float:
        if self._cosGamma is None:
            if self.validate_reciprocal_angles():
                cos_gamma = (self._cosAlphaStar * self._cosBetaStar - self._cosGammaStar)
                cos_gamma /= (self._cosAlphaStar * self._cosBetaStar)
                self._cosBeta = cos_gamma
                return cos_gamma
        else:
            return self._cosGamma

    @property
    def m_max(self) -> int:
        return self._mmax

    @property
    def n_max(self) -> int:
        return  self._nmax

    @m_max.setter
    def m_max(self, val):
        val = int(val)
        if val == 0:
            raise ValueError('Setting m_max to zero!')
        elif val < 0:
            raise Warning('Setting m_max to {0:d}. Taking the absolute value.'.format(val))
        self._mmax = abs(val)

    @n_max.setter
    def n_max(self, val):
        val = int(val)
        if val == 0:
            raise ValueError('Setting n_max to zero!')
        elif val < 0:
            raise Warning('Setting n_max to {0:d}. Taking the absolute value.'.format(val))
        self._nmax = abs(val)

    def validate_reciprocal_angles(self):
        if self._cosAlphaStar is None:
            raise errors.InvalidClassProperty(property='cos(alpha*)', value=self._cosAlphaStar)
        if self._cosBetaStar is None:
            raise errors.InvalidClassProperty(property='cos(beta*)', value=self._cosBetaStar)
        if self._cosGammaStar is None:
            raise errors.InvalidClassProperty(property='cos(gamma*)', value=self._cosGammaStar)
        return True

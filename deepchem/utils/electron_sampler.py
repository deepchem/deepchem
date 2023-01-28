import numpy as np
from typing import Optional, Callable


class ElectronSampler:
    """
    This class enables to initialize electron's position using gauss distribution around a nucleus and update using Markov Chain Monte-Carlo(MCMC) moves.

    Using the probability obtained from the square of magnitude of wavefunction of a molecule/atom, MCMC steps can be performed to get the electron's positions and further update the wavefunction.
    This method is primarily used in methods like Variational Monte Carlo to sample electrons around the nucleons.
    Sampling can be done in 2 ways:
    -Simultaneous: All the electrons' positions are updated all at once.

    -Single-electron: MCMC steps are performed only a particular electron, given their index value.

    Further these moves can be done in 2 methods:
    -Symmetric: In this configuration, the standard deviation for all the steps are uniform.

    -Asymmetric: In this configuration, the standard deviation are not uniform and typically the standard deviation is obtained a function like harmonic distances, etc.

    Irrespective of these methods, the initialization is done uniformly around the respective nucleus and the number of electrons specified.

    Example
    -------
    >>> from deepchem.utils.electron_sampler import ElectronSampler
    >>> test_f = lambda x: 2*np.log(np.random.uniform(low=0,high=1.0,size=np.shape(x)[0]))
    >>> distribution=ElectronSampler(central_value=np.array([[1,1,3],[3,2,3]]),f=test_f,seed=0,batch_no=2,steps=1000,)
    >>> distribution.gauss_initialize_position(np.array([[1],[2]]))

    >> print(distribution.x)
    [[[[1.03528105 1.00800314 3.01957476]]

      [[3.01900177 1.99697286 2.99793562]]

      [[3.00821197 2.00288087 3.02908547]]]


    [[[1.04481786 1.03735116 2.98045444]]

      [[3.01522075 2.0024335  3.00887726]]

      [[3.00667349 2.02988158 2.99589683]]]]
    >>> distribution.move()
    0.5115

    >> print(distribution.x)
    [[[[-0.32441754  1.23330263  2.67927645]]

      [[ 3.42250997  2.23617126  3.55806632]]

      [[ 3.37491385  1.54374006  3.13575241]]]


    [[[ 0.49067726  1.03987841  3.70277884]]

      [[ 3.5631939   1.68703947  2.5685874 ]]

      [[ 2.84560249  1.73998364  3.41274181]]]]
    """

    def __init__(self,
                 central_value: np.ndarray,
                 f: Callable[[np.ndarray], np.ndarray],
                 batch_no: int = 10,
                 x: np.ndarray = np.array([]),
                 steps: int = 10,
                 seed: Optional[int] = None,
                 symmetric: bool = True,
                 simultaneous: bool = True):
        """
        Parameters:
        -----------
        central_value: np.ndarray
            Contains each nucleus' coordinates in a 2D array. The shape of the array should be(number_of_nucleus,3).Ex: [[1,2,3],[3,4,5],..]
        f:Callable[[np.ndarray],np.ndarray]
            A function that should give the twice the log probability of wavefunction of the molecular system when called. Should taken in a 4D array of electron's positions(x) as argument and return a numpy array containing the log probabilities of each batch.
        batch_no: int, optional (default 10)
            Number of batches of the electron's positions to be initialized.
        x: np.ndarray, optional (default np.ndarray([]))
            Contains the electron's coordinates in a 4D array. The shape of the array should be(batch_no,no_of_electrons,1,3). Can be a 1D empty array, when electron's positions are yet to be initialized.
        steps: int, optional (default 10)
            The number of MCMC steps to be performed when the moves are called.
        seed: int, optional (default None)
            Random seed to use.
        symmetric: bool, optional(default True)
            If true, symmetric moves will be used, else asymmetric moves will be followed.
        simultaneous: bool, optional(default True)
            If true, MCMC steps will be performed on all the electrons, else only a single electron gets updated.
        """
        self.x = x
        self.f = f
        self.num_accept = 0
        self.symmetric = symmetric
        self.simultaneous = simultaneous
        self.steps = steps
        self.central_value = central_value
        self.batch_no = batch_no
        if seed is not None:
            seed = int(seed)
            np.random.seed(seed)

    def harmonic_mean(self, y: np.ndarray) -> np.ndarray:
        """Calculates the harmonic mean of the value 'y' from the self.central value. The numpy array returned is typically scaled up to get the standard deviation matrix.

        Parameters
        ----------
        y: np.ndarray
            Containing the data distribution. Shape of y should be (batch,no_of_electron,1,3)

        Return
        ----------
        np.ndarray
            Contains the harmonic mean of the data distribution of each batch. Shape of the array obtained (batch_no, no_of_electrons,1,1)
        """

        diff = y - self.central_value
        distance = np.linalg.norm(diff, axis=-1, keepdims=True)
        return 1.0 / np.mean(1.0 / distance, axis=-2, keepdims=True)

    def log_prob_gaussian(self, y: np.ndarray, mu: np.ndarray,
                          sigma: np.ndarray) -> np.ndarray:
        """Calculates the log probability of a gaussian distribution, given the mean and standard deviation

        Parameters
        ----------
        y: np.ndarray
            data for which the log normal distribution is to be found
        mu: np.ndarray
            Means wrt which the log normal is calculated. Same shape as x or should be brodcastable to x
        sigma: np.ndarray,
            The standard deviation of the log normal distribution. Same shape as x or should be brodcastable to x

        Return
        ----------
        np.ndarray
            Log probability of gaussian distribution, with the shape - (batch_no,).
        """

        numer = np.sum((-0.5 * ((y - mu)**2) / (sigma**2)), axis=(1, 2, 3))
        denom = y.shape[-1] * np.sum(np.log(sigma), axis=(1, 2, 3))
        return numer - denom

    def gauss_initialize_position(self,
                                  no_sample: np.ndarray,
                                  stddev: float = 0.02):
        """Initializes the position around a central value as mean sampled from a gauss distribution and updates self.x.
        Parameters:
        ----------
        no_sample: np.ndarray,
            Contains the number of samples to initialize under each mean. should be in the form [[3],[2]..], where here it means 3 samples and 2 samples around the first entry and second entry,respectively in self.central_value is taken.
        stddev: float, optional (default 0.02)
            contains the stddev with which the electrons' coordinates are initialized
        """

        mean = self.central_value[0]
        specific_sample = no_sample[0][0]
        ndim = np.shape(self.central_value)[1]
        self.x = np.random.normal(mean, stddev,
                                  (self.batch_no, specific_sample, 1, ndim))

        end = np.shape(self.central_value)[0]
        for i in range(1, end):
            mean = self.central_value[i]
            specific_sample = no_sample[i][0]
            self.x = np.append(
                self.x,
                np.random.normal(mean, stddev,
                                 (self.batch_no, specific_sample, 1, ndim)),
                axis=1)

    def move(self,
             stddev: float = 0.02,
             asymmetric_func: Optional[Callable[[np.ndarray],
                                                np.ndarray]] = None,
             index: Optional[int] = None) -> float:
        """Performs Metropolis-Hasting move for self.x(electrons). The type of moves to be followed -(simultaneous or single-electron, symmetric or asymmetric) have been specified when calling the class.
        The self.x array is replaced with a new array at the end of each step containing the new electron's positions.

        Parameters:
        -----------
        asymmetric_func: Callable[[np.ndarray],np.ndarray], optional(default None)
            Should be specified for an asymmetric move.The function should take in only 1 argument- y: a numpy array wrt to which mean should be calculated.
            This function should return the mean for the asymmetric proposal. For ferminet, this function is the harmonic mean of the distance between the electron and the nucleus.
        stddev: float, optional (default 0.02)
            Specifies the standard deviation in the case of symmetric moves and the scaling factor of the standard deviation matrix in the case of asymmetric moves.
        index: int, optional (default None)
            Specifies the index of the electron to be updated in the case of a single electron move.

        Return
        ------
        float
            accepted move ratio of the MCMC steps.
        """

        lp1 = self.f(self.x)  # log probability of self.x state

        if self.simultaneous:
            if self.symmetric:
                for i in range(self.steps):
                    x2 = np.random.normal(self.x, stddev, self.x.shape)
                    lp2 = self.f(x2)  # log probability of x2 state
                    ratio = lp2 - lp1
                    move_prob = np.log(
                        np.random.uniform(low=0,
                                          high=1.0,
                                          size=np.shape(self.x)[0]))
                    cond = move_prob < ratio
                    lp1 = np.where(cond, lp2, lp1)
                    self.x = np.where(cond[:, None, None, None], x2, self.x)
                    self.num_accept += np.sum(cond)

            elif asymmetric_func is not None:
                for i in range(self.steps):
                    std = stddev * asymmetric_func(self.x)
                    x2 = np.random.normal(self.x, std, self.x.shape)
                    lp2 = self.f(x2)  # log probability of x2 state
                    lq1 = self.log_prob_gaussian(self.x, x2,
                                                 std)  # forward probability
                    lq2 = self.log_prob_gaussian(
                        x2, self.x,
                        stddev * asymmetric_func(x2))  # backward probability
                    ratio = lp2 + lq2 - lq1 - lp1
                    move_prob = np.log(
                        np.random.uniform(low=0,
                                          high=1.0,
                                          size=np.shape(self.x)[0]))
                    cond = move_prob < ratio
                    self.x = np.where(cond[:, None, None, None], x2, self.x)
                    lp1 = np.where(cond, lp2, lp1)
                    self.num_accept += np.sum(cond)

        elif index is not None:
            index = int(index)
            x2 = np.copy(self.x)
            altered_shape = (self.batch_no, 1, np.shape(self.x)[3])

            if self.symmetric:
                for i in range(self.steps):
                    x2[:, index, :, :] = np.random.normal(x2[:, index, :, :],
                                                          stddev,
                                                          size=altered_shape)
                    lp2 = self.f(x2)  # log probability of x2 state
                    ratio = lp2 - lp1
                    move_prob = np.log(
                        np.random.uniform(low=0,
                                          high=1.0,
                                          size=np.shape(self.x)[0]))
                    cond = move_prob < ratio
                    lp1 = np.where(cond, lp2, lp1)
                    self.x = np.where(cond[:, None, None, None], x2, self.x)

            elif asymmetric_func is not None:
                init_dev = stddev * asymmetric_func(
                    self.x)  # initial standard deviation matrix
                for i in range(self.steps):
                    std = stddev * asymmetric_func(self.x[:, index, :, :])
                    x2[:, index, :, :] = np.random.normal(x2[:, index, :, :],
                                                          std,
                                                          size=altered_shape)
                    lp2 = self.f(x2)  # log probability of x2 state
                    init_dev[:, index, :, :] = std
                    lq1 = self.log_prob_gaussian(
                        self.x, x2, init_dev)  # forward probability
                    lq2 = self.log_prob_gaussian(
                        x2, self.x,
                        stddev * asymmetric_func(x2))  # backward probability
                    ratio = lp2 + lq2 - lq1 - lp1
                    move_prob = np.log(
                        np.random.uniform(low=0,
                                          high=1.0,
                                          size=np.shape(self.x)[0]))
                    cond = move_prob < ratio
                    self.x = np.where(cond[:, None, None, None], x2, self.x)
                    lp1 = np.where(cond, lp2, lp1)
                    self.num_accept += np.sum(cond)

        return self.num_accept / (
            (i + 1) * np.shape(self.x)[0])  # accepted move ratio

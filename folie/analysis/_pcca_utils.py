"""
Set of utilities function for PCCA++ clustering

Adapted from deeptime https://github.com/deeptime-ml/deeptime
"""


def _pcca_connected_isa(eigenvectors, n_clusters):
    """
    PCCA+ spectral clustering method using the inner simplex algorithm.

    Clusters the first n_cluster eigenvectors of a transition matrix in order to cluster the states.
    This function assumes that the state space is fully connected, i.e. the transition matrix whose
    eigenvectors are used is supposed to have only one eigenvalue 1, and the corresponding first
    eigenvector (evec[:,0]) must be constant.

    Parameters
    ----------
    eigenvectors : ndarray
        A matrix with the sorted eigenvectors in the columns. The stationary eigenvector should
        be first, then the one to the slowest relaxation process, etc.

    n_clusters : int
        Number of clusters to group to.

    Returns
    -------
    chi : ndarray (n x m)
        A matrix containing the probability or membership of each state to be assigned to each cluster.
        The rows sum to 1.

    rot_mat : ndarray (m x m)
        A rotation matrix that rotates the dominant eigenvectors to yield the PCCA memberships, i.e.:
        chi = np.dot(evec, rot_matrix

    References
    ----------
    [1] P. Deuflhard and M. Weber, Robust Perron cluster analysis in conformation dynamics.
        in: Linear Algebra Appl. 398C M. Dellnitz and S. Kirkland and M. Neumann and C. Schuette (Editors)
        Elsevier, New York, 2005, pp. 161-184

    """
    (n, m) = eigenvectors.shape

    # do we have enough eigenvectors?
    if n_clusters > m:
        raise ValueError("Cannot cluster the (" + str(n) + " x " + str(m) + " eigenvector matrix to " + str(n_clusters) + " clusters.")

    # check if the first, and only the first eigenvector is constant
    diffs = np.abs(np.max(eigenvectors, axis=0) - np.min(eigenvectors, axis=0))
    assert diffs[0] < 1e-6, "First eigenvector is not constant. This indicates that the transition matrix " "is not connected or the eigenvectors are incorrectly sorted. Cannot do PCCA."
    assert diffs[1] > 1e-6, "An eigenvector after the first one is constant. " "Probably the eigenvectors are incorrectly sorted. Cannot do PCCA."

    # local copy of the eigenvectors
    c = eigenvectors[:, list(range(n_clusters))]

    ortho_sys = np.copy(c)
    max_dist = 0.0

    # representative states
    ind = np.zeros(n_clusters, dtype=np.int32)

    # select the first representative as the most outlying point
    for i, row in enumerate(c):
        if np.linalg.norm(row, 2) > max_dist:
            max_dist = np.linalg.norm(row, 2)
            ind[0] = i

    # translate coordinates to make the first representative the origin
    ortho_sys -= c[ind[0], None]

    # select the other m-1 representatives using a Gram-Schmidt orthogonalization
    for k in range(1, n_clusters):
        max_dist = 0.0
        temp = np.copy(ortho_sys[ind[k - 1]])

        # select next farthest point that is not yet a representative
        for i, row in enumerate(ortho_sys):
            row -= np.dot(np.dot(temp, np.transpose(row)), temp)
            distt = np.linalg.norm(row, 2)
            if distt > max_dist and i not in ind[0:k]:
                max_dist = distt
                ind[k] = i
        ortho_sys /= np.linalg.norm(ortho_sys[ind[k]], 2)

    # print "Final selection ", ind

    # obtain transformation matrix of eigenvectors to membership matrix
    rot_mat = np.linalg.inv(c[ind])
    # print "Rotation matrix \n ", rot_mat

    # compute membership matrix
    chi = np.dot(c, rot_mat)
    # print "chi \n ", chi

    return chi, rot_mat


def _opt_soft(eigenvectors, rot_matrix, n_clusters):
    """
    Optimizes the PCCA+ rotation matrix such that the memberships are exclusively nonnegative.

    Parameters
    ----------
    eigenvectors : ndarray
        A matrix with the sorted eigenvectors in the columns. The stationary eigenvector should
        be first, then the one to the slowest relaxation process, etc.

    rot_matrix : ndarray (m x m)
        nonoptimized rotation matrix

    n_clusters : int
        Number of clusters to group to.

    Returns
    -------
    rot_mat : ndarray (m x m)
        Optimized rotation matrix that rotates the dominant eigenvectors to yield the PCCA memberships, i.e.:
        chi = np.dot(evec, rot_matrix

    References
    ----------
    [1] S. Roeblitz and M. Weber, Fuzzy spectral clustering by PCCA+:
        application to Markov state models and data classification.
        Adv Data Anal Classif 7, 147-179 (2013).

    """
    # only consider first n_clusters eigenvectors
    eigenvectors = eigenvectors[:, :n_clusters]

    # crop first row and first column from rot_matrix
    # rot_crop_matrix = rot_matrix[1:,1:]
    rot_crop_matrix = rot_matrix[1:][:, 1:]

    (x, y) = rot_crop_matrix.shape

    # reshape rot_crop_matrix into linear vector
    rot_crop_vec = np.reshape(rot_crop_matrix, x * y)

    # Susanna Roeblitz' target function for optimization
    def susanna_func(rot_crop_vec, eigvectors):
        # reshape into matrix
        rot_crop_matrix = np.reshape(rot_crop_vec, (x, y))
        # fill matrix
        rot_matrix = _fill_matrix(rot_crop_matrix, eigvectors)

        result = 0
        for i in range(0, n_clusters):
            for j in range(0, n_clusters):
                result += np.power(rot_matrix[j, i], 2) / rot_matrix[0, i]
        return -result

    from scipy.optimize import fmin

    rot_crop_vec_opt = fmin(susanna_func, rot_crop_vec, args=(eigenvectors,), disp=False)

    rot_crop_matrix = np.reshape(rot_crop_vec_opt, (x, y))
    rot_matrix = _fill_matrix(rot_crop_matrix, eigenvectors)

    return rot_matrix


def _fill_matrix(rot_crop_matrix, eigvectors):
    """Helper function for opt_soft"""

    (x, y) = rot_crop_matrix.shape

    row_sums = np.sum(rot_crop_matrix, axis=1)
    row_sums = np.reshape(row_sums, (x, 1))

    # add -row_sums as leftmost column to rot_crop_matrix
    rot_crop_matrix = np.concatenate((-row_sums, rot_crop_matrix), axis=1)

    tmp = -np.dot(eigvectors[:, 1:], rot_crop_matrix)

    tmp_col_max = np.max(tmp, axis=0)
    tmp_col_max = np.reshape(tmp_col_max, (1, y + 1))

    tmp_col_max_sum = np.sum(tmp_col_max)

    # add col_max as top row to rot_crop_matrix and normalize
    rot_matrix = np.concatenate((tmp_col_max, rot_crop_matrix), axis=0)
    rot_matrix /= tmp_col_max_sum

    return rot_matrix


def _pcca_connected(vecs):
    r"""PCCA+ spectral clustering method with optimized memberships [1]_

    Clusters the first n_cluster eigenvectors of a transition matrix in order to cluster the states.
    This function assumes that the transition matrix is fully connected.

    Parameters
    ----------
    P : ndarray (n,n)
        Transition matrix.
    n : int
        Number of clusters to group to.
    pi: ndarray(n,), optional, default=None
        Stationary distribution if available.

    Returns
    -------
    chi : ndarray (n x m)
        A matrix containing the probability or membership of each state to be assigned to each cluster.
        The rows sum to 1.

    References
    ----------
    [1] S. Roeblitz and M. Weber, Fuzzy spectral clustering by PCCA+:
        application to Markov state models and data classification.
        Adv Data Anal Classif 7, 147-179 (2013).
    """

    cst_vect = np.ones_like(vecs[:, 0])

    evecs = np.column_stack((cst_vect, vecs))

    n = evecs.shape[1]

    # create initial solution using PCCA+. This could have negative memberships
    chi, rot_matrix = _pcca_connected_isa(evecs, n)

    # optimize the rotation matrix with PCCA++.
    rot_matrix = _opt_soft(evecs, rot_matrix, n)

    # These memberships should be nonnegative
    memberships = np.dot(evecs[:, :], rot_matrix)

    # We might still have numerical errors. Force memberships to be in [0,1]
    memberships = np.clip(memberships, 0.0, 1.0)

    for i in range(0, np.shape(memberships)[0]):
        memberships[i] /= np.sum(memberships[i])
    return memberships

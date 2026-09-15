import numpy as np
from scipy.signal import fftconvolve
from scipy.special import logsumexp

class GridBayesFilter:
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float, delta: float,
                    nu: float = 5.0, 
                    gamma_context: float = 0.3, 
                    alpha_obs: float = 0.1,
                    beta_absent: float = 0.30):

        # Define grid parameters (size, resolution, number of cells)
        self.x_min, self.x_max = float(x_min), float(x_max)
        self.y_min, self.y_max = float(y_min), float(y_max)
        self.delta = float(delta)
        self.nx = int(np.ceil((x_max - x_min) / delta)) # Cells in x-direction (width)
        self.ny = int(np.ceil((y_max - y_min) / delta)) # Cells in y-direction (height)
        self.N = self.nx * self.ny                      # Total number of cells
        self.pi_0 = 1.0 / self.N                        # Initial probability mass for each cell (uniform prior)

        # Belief state maps (log-space and probability mass function)
        self.log_pi = np.full((self.ny, self.nx), np.log(self.pi_0), dtype=np.float64)
        self.pi = np.exp(self.log_pi)
        
        # Memorize alpha(x,y) coefficients in (0, 1] that reduce the impact of subsequent re-observations of the same 
        # semantic clue in the same spatial region (avoid to increment the pb observing the same object multiple times)
        self.alpha_maps = {} # label -> np.ndarray (ny, nx)

        # Track whether a region has been consumed by a target observation
        self.consumed = np.zeros((self.ny, self.nx), dtype=bool)

        # Track observation yaws per cell (max 4 distinct yaws stored per cell)
        self.obs_yaws_grid = np.full((self.ny, self.nx, 4), np.nan, dtype=np.float32)

        # Hyperparameters
        self.nu = float(nu)                       # Flat-Top decay in meters
        self.gamma_context = float(gamma_context) # Log-odds update strength for contexts
        self.alpha_obs = float(alpha_obs)         # Multiplicative redundancy discount factor for repeated observations
        self.beta_absent = float(beta_absent)     # Non-observation penalty for context footprint when target is absent
         
        # Grid center coordinates
        _xs = self.x_min + (np.arange(self.nx) + 0.5) * self.delta
        _ys = self.y_min + (np.arange(self.ny) + 0.5) * self.delta
        self.XX, self.YY = np.meshgrid(_xs, _ys) # store grid cell center coordinates to compute distances very fast

    def _compute_mahalanobis(self, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Compute the Mahalanobis distance for each grid cell to a given 2D Gaussian center (mu_x, mu_y)."""
        pos = np.dstack((self.XX, self.YY))
        diff = pos - mu
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.eye(2)
        D_sq = np.einsum('...i,ij,...j->...', diff, inv_cov, diff)
        return np.sqrt(np.maximum(0, D_sq))

    def _compute_flat_top(self, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Computes the flat-top kernel K_ft(x_0) in meters."""
        D_M = self._compute_mahalanobis(mu, cov)
        diff = np.dstack((self.XX - mu[0], self.YY - mu[1]))
        d_centre = np.linalg.norm(diff, axis=-1)
        
        D_M_safe = np.maximum(D_M, 1.0 + 1e-9)
        d_E = d_centre * (1.0 - 1.0 / D_M_safe)
        d_E = np.where(D_M <= 1.0, 0.0, d_E)
        
        K_ft = np.where(D_M <= 1.0, 1.0, np.exp(-(d_E**2) / (2.0 * self.nu**2)))
        return K_ft

    def _generate_spatial_kernel(self, d_star: float, sigma_c: float) -> np.ndarray:
        """Generates a Gaussian spatial kernel f(d) based on the current grid."""
        radius = int(np.ceil((d_star + 3.0 * sigma_c) / self.delta))
        size = 2 * radius + 1
        ax = (np.arange(size) - radius) * self.delta
        xx, yy = np.meshgrid(ax, ax)
        d = np.hypot(xx, yy)
        return np.exp(-((d - d_star)**2) / (2.0 * sigma_c**2))

    def _compute_positive_likelihood(self, grouped_observations: dict) -> tuple[np.ndarray, np.ndarray, dict]:
        """Calculates positive log-likelihood log L_pos(x) from VLM detections (target + contexts)."""
        log_L_pos = np.zeros((self.ny, self.nx), dtype=np.float64)
        target_observed_mask = np.zeros((self.ny, self.nx), dtype=bool)
        label_l_max = {}

        for label, obs_list in grouped_observations.items():
            if not obs_list:
                continue

            if label not in self.alpha_maps:
                self.alpha_maps[label] = np.ones((self.ny, self.nx), dtype=np.float64)

            K_ft_label = np.zeros((self.ny, self.nx), dtype=np.float64)

            is_target = False
            sigma_c = 0.0
            d_star = 0.0

            for mu, cov, sigma_json, d_star_json in obs_list:
                sigma_c = float(sigma_json)
                d_star = float(d_star_json)

                if sigma_c == 0.0:
                    is_target = True
                    sigma_c = self.delta 
                
                K_ft = self._compute_flat_top(mu, cov)
                K_ft_label = np.maximum(K_ft_label, K_ft)

            # Evidence weighted by redundancy discount
            E = self.alpha_maps[label] * K_ft_label

            if is_target:
                target_observed_mask |= (E > 0.1)
                label_l_max[label] = 1.0
                continue

            # Generative spatial kernel f(d)
            kernel_f = self._generate_spatial_kernel(d_star=d_star, sigma_c=sigma_c)
            kernel_sum = float(np.sum(kernel_f))
            kernel_f_norm = (kernel_f / kernel_sum) if kernel_sum > 0 else kernel_f
            L_conv = fftconvolve(E, kernel_f_norm, mode='same')

            # Channel log-odds scale log L_c = n_c * L_conv
            gamma = self.gamma_context
            log_L_c = gamma * L_conv

            log_L_pos += log_L_c
            label_l_max[label] = float(np.max(log_L_c))

            # Multiplicative redundancy discount for observed cells in detection footprint
            self.alpha_maps[label][K_ft_label > 0.01] = np.maximum(
                self.alpha_maps[label][K_ft_label > 0.01] * self.alpha_obs, 1e-4
            )

        return log_L_pos, target_observed_mask, label_l_max

    def update_evidence(self, grouped_observations: dict, footprint_mask: np.ndarray = None, coverage_mask: np.ndarray = None):
        """Computes the posterior update based on new observations and immediate frame-level target absence likelihood."""
        # Track total covered area from coverage map
        if coverage_mask is not None:
            self.consumed |= coverage_mask

        # 1. Positive Log-Likelihood (Detections)
        log_L_pos, target_observed_mask, label_l_max = self._compute_positive_likelihood(grouped_observations)
        target_found = bool(np.any(target_observed_mask))

        # 2. Target Absence Log-Likelihood: applied IMMEDIATELY to all cells within the current camera footprint
        log_L_absent = np.zeros((self.ny, self.nx), dtype=np.float64)
        if self.beta_absent > 0.0 and footprint_mask is not None and np.any(footprint_mask):
            log_L_absent[footprint_mask] = np.log(np.maximum(1.0 - self.beta_absent, 1e-6))

        # Total Combined Log-Likelihood
        log_L_total = log_L_pos + log_L_absent

        # Cache likelihoods for logging and visualization
        self.latest_likelihood = np.exp(log_L_total)
        self.latest_positive_likelihood = np.exp(log_L_pos)

        # 3. Recursive Bayes Update in Log Space
        log_pi_tilde = self.log_pi + log_L_total

        # 4. LogSumExp Normalization
        log_eta = logsumexp(log_pi_tilde)
        self.log_pi = log_pi_tilde - log_eta

        # Convert to probability mass function
        self.pi = np.exp(self.log_pi)

        stats = {
            "L_pos_max": float(np.max(np.exp(log_L_pos))),
            "L_pos_min": float(np.min(np.exp(log_L_pos))),
            "L_absent_min": float(np.min(np.exp(log_L_absent))),
            "target_found": target_found,
            "pi_max": float(np.max(self.pi)),
            "pi_min": float(np.min(self.pi)),
            "label_l_max": label_l_max,
            "alpha_stats": {lbl: (float(np.min(arr)), float(np.max(arr))) for lbl, arr in self.alpha_maps.items()}
        }
        return stats

    def get_latest_likelihood_normalized(self, max_scale: float = 0.5) -> np.ndarray:
        """Returns absolute total likelihood increase (L - 1.0) scaled against a fixed reference scale."""
        if not hasattr(self, 'latest_likelihood'):
            return np.zeros((self.ny, self.nx), dtype=np.float64)
        
        L = self.latest_likelihood
        delta_L = np.maximum(L - 1.0, 0.0)
        return np.clip(delta_L / max_scale, 0.0, 1.0)

    def get_latest_positive_likelihood_normalized(self, max_scale: float = 0.5) -> np.ndarray:
        """Returns absolute positive likelihood increase (L_pos - 1.0) scaled against a fixed reference scale."""
        if not hasattr(self, 'latest_positive_likelihood'):
            return np.zeros((self.ny, self.nx), dtype=np.float64)
        
        L = self.latest_positive_likelihood
        delta_L = np.maximum(L - 1.0, 0.0)
        return np.clip(delta_L / max_scale, 0.0, 1.0)

    def get_posterior_probability(self, zero_consumed: bool = False) -> np.ndarray:
        """Returns the posterior probability distribution. If zero_consumed is True, it sets the posterior probability of the seen cells to 0."""
        P = self.pi.copy()
        if zero_consumed:
            P[self.consumed] = 0.0
        return P
    
    def get_log_pi(self) -> np.ndarray:
        """Returns the log of the posterior probability distribution pi."""
        P = np.clip(self.pi.copy(), 1e-15, 1.0)
        log_pi = np.log(P)
        log_pi[self.consumed] = -np.inf
        return log_pi

    def get_log_F(self) -> np.ndarray:
        """Alias for get_log_pi()."""
        return self.get_log_pi()

    def idx_to_world(self, i: int, j: int):
        """Converts grid indices to world coordinates."""
        return (self.x_min + (j + 0.5) * self.delta, self.y_min + (i + 0.5) * self.delta)


    def record_observation_yaw(self, footprint_mask: np.ndarray, yaw: float):
        """
        Record the observation yaw angle for all cells within footprint_mask.
        Stores up to 4 distinct angles per cell (discarding angles closer than ~15 deg to an existing angle).
        """
        if footprint_mask is None or not np.any(footprint_mask):
            return

        yaw = float(yaw)
        mask_cells = np.where(footprint_mask)
        if len(mask_cells[0]) == 0:
            return

        yaws_slice = self.obs_yaws_grid[mask_cells[0], mask_cells[1], :] # shape (M, 4)
        diffs = np.abs(np.arctan2(np.sin(yaw - yaws_slice), np.cos(yaw - yaws_slice)))
        already_present = np.any(diffs < 0.26, axis=1)

        cells_to_update_idx = np.where(~already_present)[0]
        if len(cells_to_update_idx) == 0:
            return

        u_rows = mask_cells[0][cells_to_update_idx]
        u_cols = mask_cells[1][cells_to_update_idx]

        for r, c in zip(u_rows, u_cols):
            slots = self.obs_yaws_grid[r, c]
            nan_idxs = np.where(np.isnan(slots))[0]
            if len(nan_idxs) > 0:
                self.obs_yaws_grid[r, c, nan_idxs[0]] = yaw
            else:
                diffs_c = np.abs(np.arctan2(np.sin(yaw - slots), np.cos(yaw - slots)))
                self.obs_yaws_grid[r, c, np.argmin(diffs_c)] = yaw

    def compute_reobservation_gain(self, candidate_yaw: float, mask: np.ndarray, pi_slice: np.ndarray, consumed_slice: np.ndarray, i0: int, i1: int, j0: int, j1: int) -> float:
        """
        Compute re-observation information gain for candidate waypoint viewing footprint 'mask'.
        Only evaluates cells that are already consumed AND have semantic importance (pi > pi_0).
        """
        already_observed_mask = mask & consumed_slice
        if not np.any(already_observed_mask):
            return 0.0

        obs_yaws_slice = self.obs_yaws_grid[i0:i1+1, j0:j1+1, :] # shape (H, W, 4)

        # Calculate angular differences for candidate_yaw
        diffs = np.abs(np.arctan2(np.sin(candidate_yaw - obs_yaws_slice), np.cos(candidate_yaw - obs_yaws_slice)))

        # Min angular difference across recorded slots per cell
        with np.errstate(all='ignore'):
            all_nan_mask = np.all(np.isnan(obs_yaws_slice), axis=-1)
            diffs_safe = np.where(np.isnan(diffs), np.pi, diffs)
            min_diffs = np.min(diffs_safe, axis=-1)
            min_diffs[all_nan_mask] = 0.0

        # Angular diversity score g(delta_theta) = 0.5 * (1 - cos(delta_theta)) in [0, 1]
        g_theta = 0.5 * (1.0 - np.cos(min_diffs))

        # Consider only cells that are more interesting that the background noise
        excess_pi = np.maximum((pi_slice / self.pi_0)-1.0, 0.0)

        # Weight re-observation gain by posterior probability mass pi_slice
        ig_reobs = float(np.sum(excess_pi[already_observed_mask] * g_theta[already_observed_mask]))
        return ig_reobs

    def reset(self):
        """Resets the filter to its initial state."""
        self.pi.fill(1.0)
        self.pi /= np.sum(self.pi)
        self.alpha_maps.clear()
        self.consumed.fill(False)
        self.obs_yaws_grid.fill(np.nan)


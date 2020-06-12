import numpy as np
from scipy.sparse import csr_matrix


class GridMatrixHelper:
    def __init__(self, alpha, beta, cache=False):
        self.inv_cov_mtx = {}
        self.cov_mtx = {}
        self.bd_cov_mtx = {}
        self.cache = cache
        self.alpha = alpha
        self.beta = beta

    # for an inner grid of size m, n, iterate all possible slice actions
    # for m = n = 1 the (1,0) and (0,1) actions produce the same outcome but
    # that's fine for now
    def iter_slice_actions(self, m, n):
        for i in range(1, m+1):
            yield (i, 0)

        for j in range(1, n+1):
            yield (0, j)


    # use formula for conditional expectation to get mean and covariance
    # of a set of variables conditioned on the value that variables on the
    # grid boundary take
    def get_mean_and_cov(self, m, n, bd_vals, bd_var_order, inner_vars):
        # m,n -- grid size specifier.  think of this as the inner part 
        #        of a grid sized m+2 x n+2
        # bd_vals -- the observed values that the points on the boundary
        #            of the m+2 x n+2 grid took
        # bd_var_order -- specifies the order of variables in the bd_vals list
        # inner_vars --- list of variables (of the form (i, j)) for which you want
        #                mean & cov
        inv_cov_mtx = self.get_inv_cov_mtx(m+2, n+2)
        inv_cov_var_map = self.get_var_idx_map(m+2,n+2)

        # generate Q_{UU} and Q_{US}, where Q = precision matrix
        Q_UU_list = []
        Q_US_list = []
        for row_var in inner_vars:
            row_list = []
            row_ind = inv_cov_var_map[row_var]
            for col_var in inner_vars:
                col_ind = inv_cov_var_map[col_var]
                row_list.append(inv_cov_mtx[row_ind, col_ind])
            Q_UU_list.append(row_list)

            row_list = []
            for col_var in bd_var_order:
                col_ind = inv_cov_var_map[col_var]
                row_list.append(inv_cov_mtx[row_ind, col_ind])
            Q_US_list.append(row_list)

        Q_UU = np.matrix(Q_UU_list)
        Q_US = np.matrix(Q_US_list)
        X_S = np.array(bd_vals)

        Q_UU_inv = np.linalg.inv(Q_UU)

        # weird fiddling with numpy types....

        mean = np.matmul(-Q_UU * Q_US, X_S)

        mean = np.array(mean).flatten()

        return mean, Q_UU_inv

    
    def get_slice_action_subgrids(self, m, n, slice_type):
        #return pairs of subgrids

        if slice_type[1] == 0:
            # row slice action
            assert(slice_type[0] >= 1 and slice_type[0] <= m)

            left_rows = slice_type[0] - 1
            right_rows = m - slice_type[0] 

            if left_rows == 0:
                left_subgrid = None
            else:
                left_subgrid = (left_rows, n)

            if right_rows == 0:
                right_subgrid = None
            else:
                right_subgrid = (right_rows, n)
        elif slice_type[0] == 0:

            # col slice action
            assert(slice_type[1] >= 1 and slice_type[1] <= n)

            left_cols = slice_type[1] - 1
            right_cols = n - slice_type[1]

            if left_cols == 0:
                left_subgrid = None
            else:
                left_subgrid = (m, left_cols)

            if right_cols == 0:
                right_subgrid = None
            else:
                right_subgrid = (m, right_cols)

        else:
            raise(Exception("invalid slice_action"))
        
        return left_subgrid, right_subgrid


    def find_closest_state(self, sample, approx_state_vals):
        best_val = -1
        best_appx_state = None
        for appx_state in approx_state_vals:
            if best_val == -1 or np.linalg.norm(np.array(sample) - np.array(appx_state)) < best_val:
                best_val = np.linalg.norm(np.array(sample) - np.array(appx_state))
                best_appx_state = appx_state
        return best_appx_state


    def compute_subgrid_bd_vals(self, maingrid, subgrid_a,
            subgrid_b, slice_action, slice_sample, main_bd_vals, bd_var_order_map):

        m, n = maingrid

        slice_var_order = self.get_slice_action_var_order(m, n, slice_action)
        slice_var_lookup = {svar:idx for idx, svar in enumerate(slice_var_order)}

        main_bd_order = bd_var_order_map[maingrid]
        maingrid_bd_lookup = {bd_var:idx for idx, bd_var in enumerate(main_bd_order)}

        if subgrid_a != None:
            subgrid_a_bd_sample = []
            subgrid_a_bd_var_order = bd_var_order_map[subgrid_a]

            for bd_var in subgrid_a_bd_var_order:
                # no transformation for non-slice boundary vars needed for subgrid a
                if bd_var in maingrid_bd_lookup:
                    subgrid_a_bd_sample.append(main_bd_vals[maingrid_bd_lookup[bd_var]])
                else:
                    assert(bd_var in slice_var_lookup)
                    subgrid_a_bd_sample.append(slice_sample[slice_var_lookup[bd_var]])
                    # if the bd-var is not in the main grid boundary then it has to come
                    # from the slice
                    #if slice_action[1] == 0:
                    #    assert(bd_var[0] == slice_action[0])
                    #    ...
                    #else:
                    #    assert(bd_var[1] == slice_action[1])
            
        else:
            subgrid_a_bd_sample = None

        if subgrid_b != None:
            subgrid_b_bd_sample = []
            subgrid_b_bd_var_order = bd_var_order_map[subgrid_b]
            
            for bd_var in subgrid_b_bd_var_order:
                # have to appropriately transform subgrid b boundary vars.
                if slice_action[1] == 0:
                    # row slice 
                    bd_var = (bd_var[0] + slice_action[0], bd_var[1])

                elif slice_action[0] == 0:
                    # column slice
                    bd_var = (bd_var[0], bd_var[1] + slice_action[1])
                else:
                    raise(Exception("wrong slice action type"))

                if bd_var in maingrid_bd_lookup:
                    subgrid_b_bd_sample.append(main_bd_vals[maingrid_bd_lookup[bd_var]])
                else:
                    if bd_var not in slice_var_lookup:
                        import pdb; pdb.set_trace()
                    assert(bd_var in slice_var_lookup)
                    subgrid_b_bd_sample.append(slice_sample[slice_var_lookup[bd_var]])
        else:
            subgrid_b_bd_sample = None

        return subgrid_a_bd_sample, subgrid_b_bd_sample
   

    
    def get_slice_action_var_order(self, m, n, slice_type):
        if slice_type[1] == 0:
            assert(slice_type[0] >= 1 and slice_type[0] <= m)
            slice_vars = [(slice_type[0], j) for j in range(1, n+1)]
        elif slice_type[0] == 0:
            assert(slice_type[1] >= 1 and slice_type[1] <= n)
            slice_vars = [(i, slice_type[1]) for i in range(1, m+1)]
        else:
            raise(Exception("invalid slice_type"))
        return slice_vars

    # iterate over sever samples of a particular choice of "slice" observation
    def iter_slice_samples(self, m, n, bd_vals, bd_var_order, slice_type, nreps):
        # m,n -- grid size specifier.  think of this as the inner part 
        #        of a grid sized m+2 x n+2
        # bd_vals -- the observed values that the points on the boundary
        #            of the m+2 x n+2 grid took
        # bd_var_order -- specifies the order of variables in the bd_vals list
        # slice_type -- either (i, 0) or (0, j) with 1 <= i <= m or 1 <= j <= n.
        #               specifies which row or column slice to consider
        # nreps --- how many samples to generate

        slice_vars = self.get_slice_action_var_order(m,n,slice_type)
        
        slice_mean, slice_cov = self.get_mean_and_cov(m, n, bd_vals, bd_var_order, slice_vars)

        samples = np.random.multivariate_normal(slice_mean, slice_cov, nreps)

        for idx in range(nreps):
            #import pdb; pdb.set_trace()
            yield samples[idx,:]

    def get_bd_vars(self, m, n):
        # get a list of variables on the boundary of an
        # m+2 x n+2 grid in a consistent order
        bd_vars = []
        
        # top boundary; i = 0; j = 1...n
        i = 0
        for j in range(1, n+1):
            bd_vars.append((i,j))
        
        # right boundy; j = n+1; i = 1..m
        j = n+1
        for i in range(1,m+1):
            bd_vars.append((i,j))
        
        # bottom boundary; i = m+1; j = n..1
        i = m+1
        for diff_j in range(1,n+1):
            j = n+1 - diff_j
            bd_vars.append((i,j))
        
        # left boundy; j = 0; i = m..1
        j = 0
        for diff_i in range(1, m+1):
            i = m + 1 - diff_i
            bd_vars.append((i,j))
        return bd_vars

    
    # get the covariance matrix of the outer-boundary 
    # of an m x n grid.
    def get_bd_cov_mtx(self, m, n):
        # for the outer-boundary need to add a row / column on
        # either side
        cov_mtx = self.get_cov_mtx(m+2,n+2)
        var_map = self.get_var_idx_map(m+2, n+2)
        
        
        bd_vars = self.get_bd_vars(m, n)
               
        bd_cov = []
        for row_var in bd_vars:
            row = []
            row_ind = var_map[row_var]
            for col_var in bd_vars:
                col_ind = var_map[col_var]
                row.append(cov_mtx[row_ind, col_ind])
            bd_cov.append(row)
        return np.matrix(bd_cov), bd_vars
        
        
    def get_cov_mtx(self, m, n):
        if (m,n) in self.cov_mtx:
            return self.cov_mtx[(m,n)]
        inv_cov = self.get_inv_cov_mtx(m,n)
        mtx = np.linalg.inv(inv_cov)
        if self.cache:
            self.cov_mtx[(m,n)] = mtx
        return mtx
    
    # generate the inv-cov matrix A = alpha I + (1-alpha) * beta * L,
    # where L is the laplacian matrix for the m x n grid
    def get_inv_cov_mtx(self,m,n):

        if (m,n) in self.inv_cov_mtx:
            return self.inv_cov_mtx[(m,n)]
        
        var_map = self.get_var_idx_map(m,n)
        inv_cov_data = []
        row_ind = []
        col_ind = []
        for (i,j) in var_map:
            row = var_map[(i,j)]
            
            if i == 0 or i == m - 1 or j == 0 or j == n - 1:
                if (i == 0 or i == m - 1) and (j == 0 or j == n - 1):
                    degree = 2
                else:
                    degree = 3
            else:
                degree = 4
            inv_cov_data.append(degree * self.alpha * self.beta + self.alpha)
            row_ind.append(row)
            col_ind.append(row)
            
            if (i,j+1) in var_map:
                col = var_map[(i,j+1)]
                inv_cov_data.append(-self.alpha * self.beta)
                row_ind.append(row)
                col_ind.append(col)
            
            if (i,j-1) in var_map:
                col = var_map[(i,j-1)]
                inv_cov_data.append(-self.alpha * self.beta)
                row_ind.append(row)
                col_ind.append(col)
            
            if (i+1,j) in var_map:
                col = var_map[(i+1,j)]
                inv_cov_data.append(-self.alpha * self.beta)
                row_ind.append(row)
                col_ind.append(col)
                
            if (i-1,j) in var_map:
                col = var_map[(i-1,j)]
                inv_cov_data.append(-self.alpha * self.beta)
                row_ind.append(row)
                col_ind.append(col)
                
        mtx = csr_matrix((inv_cov_data, (row_ind, col_ind))).todense()
        if self.cache:
            self.inv_cov_mtx[(m,n)] = mtx
        return mtx
        
    def get_var_idx_map(self, m, n):
        var_map = {}
        idx = 0
        for i in range(m):
            for j in range(n):
                var_map[(i,j)] = idx
                idx += 1
        return var_map

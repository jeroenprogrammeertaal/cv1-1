import numpy as np

def construct_surface(p, q, path_type='column'):

    '''
    CONSTRUCT_SURFACE construct the surface function represented as height_map
       p : measures value of df / dx
       q : measures value of df / dy
       path_type: type of path to construct height_map, either 'column',
       'row', or 'average'
       height_map: the reconstructed surface
    '''
    
    h, w = p.shape
    height_map = np.zeros([h, w])
    
    if path_type=='column':
        """
        ================
        Your code here
        ================
        % top left corner of height_map is zero
        % for each pixel in the left column of height_map
        %   height_value = previous_height_value + corresponding_q_value
        
        % for each row
        %   for each element of the row except for leftmost
        %       height_value = previous_height_value + corresponding_p_value
        
        """
        for idx in range(1, h):
            height_map[idx, 0] = height_map[idx-1, 0] + q[idx, 0]

        for row_idx in range(h):
            for col_idx in range(1, w):
                height_map[row_idx, col_idx] = height_map[row_idx, col_idx - 1] + p[row_idx, col_idx]            

    elif path_type=='row':
        """
        ================
        Your code here
        ================
        """
        for idx in range(1, w):
            height_map[0, idx] = height_map[0, idx - 1] + p[0, idx]

        for col_idx in range(w):
            for row_idx in range(1, h):
                height_map[row_idx, col_idx] = height_map[row_idx - 1, col_idx] + q[row_idx, col_idx]

    elif path_type=='average':
        """
        ================
        Your code here
        ================
        """
        col_path = construct_surface(p, q, path_type='column')
        row_path = construct_surface(p, q, path_type='row')

        height_map = (col_path + row_path) / 2

        
    return height_map
        

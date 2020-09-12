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
        height_map[0,0] = 0
        prev_height_value = 0.0
        for i in range(h):
            height_map[i,0] = prev_height_value + q[i,0]
            prev_height_value = height_map[i,0]
        
        for i in range(h):
           prev_height_value = height_map[i,0]
           for j in range(1, w):
              height_map[i,j] = prev_height_value + p[i,j]
              prev_height_value = height_map[i,j]
            


    elif path_type=='row':
        """
        ================
        Your code here
        ================
        """
    elif path_type=='average':
        """
        ================
        Your code here
        ================
        """
        
    return height_map
        

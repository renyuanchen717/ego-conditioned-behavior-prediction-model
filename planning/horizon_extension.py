import numpy as np
def extend_boxes(coords):
    assert coords.shape==(5,81,2),coords.shape
    displacement=coords[4,-1]-coords[4,-2]
    extra=coords[:,-1:, :]+np.arange(1,11)[None,:,None]*displacement
    result=np.concatenate([coords,extra],axis=1)
    assert result.shape==(5,91,2)
    np.testing.assert_array_equal(result[:,:81],coords)
    return result

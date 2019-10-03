'''
Functions to go in here (I think!?):
    KC: 01/12/2018, ideas-
    ~

    KC: 28/03/2019, added-
    ~find_my_y()
'''
#no dependencies

'''
Alterations:
    KC: 29/03/2019 - changed find_my_x() to find_my_y().
'''

#interpolation function
def find_my_y(your_x, data_x, data_y, logged_data=False):
    """Takes an input x, linear interpolates the data and produces a corresponding y(s).
    
    Parameters
    ----------
    your_x : float
            A single number, of which you want the corresponding y value(s) of through linear interpolation of the data 
            given (data_x, data_y).
    
    data_x : 1-d list/array
            This is the original set of x values.
            
    data_y : 1-d list/array
            This is the original set of y values.
            
    logged_data : Bool
            If the data is logged coming in and you want linear values back out set this to True.
            Default: False
            
    Returns
    -------
    A list of corresponding y(s) to the input your_x.
    """
    
    your_x_between = []
    #search for y range which has your point in it
    for dx in range(len(data_x)-1):
        if dx == 0: #so the first one isnt completely discounted
            if data_x[dx] <= your_x <= data_x[dx+1]:
                #append the coordinates of the range
                your_x_between.append([[data_x[dx], data_y[dx]], [data_x[dx+1], data_y[dx+1]]])
        else:
            if (data_x[dx] < your_x <= data_x[dx+1]) or (data_x[dx] > your_x >= data_x[dx+1]):
                your_x_between.append([[data_x[dx], data_y[dx]], [data_x[dx+1], data_y[dx+1]]])
            
    #no extrapolation, if your_x is not within the set of x's given (data_x) then this won't work    
    if your_x_between == []: 
        print('Your x is out of range of this data_x.')
        return
    
    #make a straight line betwen the points and plug your x value in
    found_y = []
    for coords in your_x_between:
        coord1 = coords[0]
        coord2 = coords[1]
        grad = (coord1[1] - coord2[1]) / (coord1[0] - coord2[0])
        _found_y = grad * (your_x - coord1[0]) + coord1[1]
        found_y.append(_found_y)
    
    #return all the y's found, no guarentee the there is a one-to-one mapping
    if logged_data == True:
        return [10**y for y in found_y]
    else:
        return found_y

'''
Functions to go in here (I think!?):
    KC: 01/12/2018, ideas-
    ~

    KC: 28/03/2019, added-
    ~find_my_x()
'''
#no dependencies

'''
Alterations:
    KC: **/**/**** - .
'''

def find_my_x(your_y, data_x, data_y):
    """Takes an input y, linear interpolates the data and produces a corresponding x.
    
    Parameters
    ----------
    your_y : float
            A single number, of which you want the corresponding x value(s) of through linear interpolation of the data 
            given (data_x, data_y).
    
    data_x : 1-d list/array
            This is the original set of x values.
            
    data_y : 1-d list/array
            This is the original set of y values.
            
    Returns
    -------
    A list of corresponding x(s) to the input your_y.
    """
    
    your_y_between = []
    #search for y range which has your point in it
    for dy in range(len(data_y)-1):
        if dy == 0: #so the first one isnt completely discounted
            if data_y[dy] <= your_y <= data_y[dy+1]:
                #append the coordinates of the range
                your_y_between.append([[data_x[dy], data_y[dy]], [data_x[dy+1], data_y[dy+1]]])
        else:
            if (data_y[dy] < your_y <= data_y[dy+1]) or (data_y[dy] > your_y >= data_y[dy+1]):
                your_y_between.append([[data_x[dy], data_y[dy]], [data_x[dy+1], data_y[dy+1]]])
            
    #no extrapolation, if your_y is not within the set of y's given (data_y) then this won't work    
    if your_y_between != []: 
        print('Your y is out of range of this data_y.')
        return
    
    #make a straight line betwen the points and plug your y value in
    found_x = []
    for coords in your_y_between:
        coord1 = coords[0]
        coord2 = coords[1]
        grad = (coord1[1] - coord2[1]) / (coord1[0] - coord2[0])
        _found_x = ( (your_y - coord1[1]) / grad) + coord1[0]
        found_x.append(_found_x)
    
    #return all the x's found, no guarentee the there is a one-to-one mapping
    return found_x

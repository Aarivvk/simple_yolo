## Writing the loss function

Loss function for each cell, 
- each cell can have one object
- each cell can have K number of anchors
    - each anchors will have width abd height
- each cell will predict the object class 
- each cell will predict the objectness
- each cell will predict the x,y and height and width


the error for the bounding box would be :
1. calculating the location information
    predicted value would be (x,y), to this apply the sigmoid which gives value in [0,1]
    now to calculate the location in the image add the cell which it predicted (Cx,Cy)
    Bx = sigm(x) + Cx
    By = sigm(y) + Cy

    root mean squared error(Bx, Tx)

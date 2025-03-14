# TODO

build in xyzidb- create obj id for the mask- make sure that the mask is with correct obj id- add key for mask_obj_id in return
modify code for mutiple objects id not just for 2


# Done

in src\dataloader\bop_base.py 
    we have listed all the call-in functions in class we need to build for each class

src\models\maskrcnn.py
    we use the static_method to call model
    also use preperties to call model in diff way- nicer just model.mask_rcnn then have it like u r calling an attrubute- nice APi - advatage of properties




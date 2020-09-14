#### CV1 Assignment 1: Photometric Stereo and Color (Submitted by Team 52A)
* Jeroen Taal 11755075
* Rajeev Verma 13250507

##### Instructions to run the code
###### 1. Photometric Stereo
* In `photometric_stereo.py`, set the path to the desired dataset folder, and run `python photometric_stereo.py`. 
* To use the shadow trick: make sure is `shadow_trick=True` in `estimate_alb_nrm (image_stack, scriptV, shadow_trick=True) function call [line 24, photometric_stereo.py]`. We experiment with three paths `column, row, average` to construct the height map in the algorithm. Vary `path_type` arg in the `construct_surface(p, q, path_type='column') function call [line  37, photometric_stereo.py]`.
* Please use the added MonkeyGraySmall for the figures of question 1.4
###### 2. Color Spaces
* In `ConvertColourSpace.py`: provide the algorithm as input for the ConvertColourSpace function. Then run `python convertcolourspace.py`
###### 3. Intrinsic Image Decomposition
* Reconstruction
           `python iid_image_formation.py ./ball.png ./ball_albedo.png ./ball_shading.png`
* Recoloring
           `python recoloring.py ./ball.png ./ball_albedo.png ./ball_shading.png`
###### 4. Color Constancy `python AWB.py`
* Run `python AWB.py`

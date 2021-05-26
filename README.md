# Digital-Twin-Bike
Digital twin of a bike


In the GitHub folder there are 4 .py files. The three of these files are the individual files making up the general code. The file, final.py
includes the joint_angles_v2.py, vibrations_final.py and route.py.

- joint_angles_v2.py - This aspect of the program defines a geometrical model of the bike, it takes into consideration the input height of the rider
			and it calculates the knee and hip angles at the two most important pedal positions. 

- vibrations_final.py - This part of the program creates the analytical vibrational model for the bike, the mechanical properties are inputted and 
			the user can then run the model and check the response of the model to the input vibrations. The code principally solves
			the equation of motion of the 8-dof system, the equation of motion is that for a stiff damped and forced lumped mass system. 
			A road profile generator is also included in this file, the generator creates a road profile based on ISO standards by using
			a sin approximation method.

- route.py - The route - This file creates the grounds for simulation, in fact the routes are the inputs to the simulation as these dictate the length
			 and the elevation of the journey.

- final.py - Within this file the above are included, and, moreover, the performance aspect of the model is provided. The performance model solve the 
		equation of motion of the bike, this equation includes a variety of forces present in the physical activity of bike riding, for example: 
		aerodynamic resistance, wheel bearing friction, rolling resistance, and also the work needed to overcomethe change of potential and kinetic
		 energy. 


To run the model,download all the finals and put them on the same folder. Run final.py. Make sure you have all the necesary libraries to run the code. A GUI will appear, this will give the user a wide range of options to select, the options are the inputs can be regarded as the settings of a study. Once the settings are chosen the study can be initiated, the program will take up to 1 minute to solve, once this is done the solutions will appear on the GUI as either numerical values or graphs. Run function (Run constant power) until you have some results that you want to plot onto the GUI and run function (Plot graphs).

To run another study the interface settings can be re-inputted and the solutions will be over-written.

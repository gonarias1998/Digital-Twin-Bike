import numpy as np
import math as mth
from math import sin
from math import cos
from scipy.linalg import eigh
from numpy.linalg import inv
from matplotlib import pyplot as plt


def joint_angle(height,crank):
    #coordinates of pedal crank
    #evantually move around the circle
    #created with centre point pedal axel
    x_crank = np.zeros((360,1))
    y_crank = np.zeros((360,1))

    x_foot = np.zeros((360,1))
    y_foot = np.zeros((360,1))


    ##################################
    # Height Variable and crak length
    p_height = height  #in mm
    crank_length = crank#in mm 
    ##################################

    x_stature1 = 1625  #5th
    #x_stature2 = 1740
    x_stature2 = 1855  #95th


    foot_length_i = 180 + (p_height - x_stature1)*((225-180)/(x_stature2 - x_stature1))+25
    knee_height_17_i = 495 + (p_height - x_stature1)*((595-495)/(x_stature2 - x_stature1))
    popliteal_heigth_18_i = 395 + (p_height - x_stature1)*((490-395)/(x_stature2 - x_stature1))-25
    thigh_thickness_14_i = 135 + (p_height - x_stature1)*((185-135)/(x_stature2 - x_stature1))
    buttock_popliteal_length_16_i = 440 + (p_height - x_stature1)*((550-440)/(x_stature2 - x_stature1))
    buttock_knee_length_15_i = 545 + (p_height - x_stature1)*((645-545)/(x_stature2 - x_stature1))
    HipHeight_i = 840 + (p_height - x_stature1)*((920-840)/(x_stature2 - x_stature1))


    foot_angle = 15*(mth.pi/180)
    leg_length = popliteal_heigth_18_i + (knee_height_17_i - popliteal_heigth_18_i)/2  
    thigh_length = buttock_popliteal_length_16_i + ((buttock_knee_length_15_i-buttock_popliteal_length_16_i)/2) - (thigh_thickness_14_i/2)
    SeatPostAngle = 74*(mth.pi/180)

    ############################
    #############################
    ##############################
    #coordinates
    x_origin = 200
    y_origin = 100

    x_torso = 16.5 + 442.17 
    y_torso = 740 + 398.13  



    g1 = ((HipHeight_i*1.105) - 200)/sin(SeatPostAngle)
    g2 = (HipHeight_i*1.105) - 200
    g = round((g1**2 -g2**2)**0.5, 1)

    #coordinates of saddle point
    x_saddle = x_origin - g 
    y_saddle = g2 + y_origin


    #Crank = (x_crank, y_crank)


    crank_angle = round(360*mth.pi/180, 4)
    crank_angle_step = round(1*mth.pi/180, 4)

    P1 = []
    P2 = []
    S1 = []
    S2 = []
    for t in np.arange(0, crank_angle, crank_angle_step):
        x_crank_new = 200 + (crank_length*cos(t))
        y_crank_new = 100 + (crank_length*sin(t)) 
        x_saddle_new = x_saddle + (thigh_length*cos(t))
        y_saddle_new = y_saddle + (thigh_length*sin(t))
        x_crank = x_crank_new 
        y_crank = y_crank_new
        x_saddle1 = x_saddle_new
        y_saddle1 = y_saddle_new
        P1.append(x_crank)
        P2.append(y_crank)
        S1.append(x_saddle1)
        S2.append(y_saddle1)



    F1 = []
    F2 = []
    #FOOT POSITION MODELLING
    for t in np.arange(0, 360, 1):
        x_foot_new = P1[t] - foot_length_i + 40
        y_foot_new = P2[t] + (sin(foot_angle)*foot_length_i)
        x_foot = x_foot_new
        y_foot = y_foot_new
        F1.append(x_foot)
        F2.append(y_foot)


    D1 =[]
    #distance between the circle built on the heel point
    #and the circle built with centre at the saddle point
    for t in np.arange(0, 360,1):
        Distance_new = ((F1[t] - x_saddle)**2 + (F2[t] - y_saddle)**2)**0.5
        Distance = Distance_new
        D1.append(Distance)


    #AREA of the triangle formed by the two circle 
    #centers and one of the intersection point.
    L1 = []
    for t in np.arange(0, 360,1):
        lamda_new =  0.25*(((D1[t] + thigh_length + leg_length)*(D1[t] + thigh_length - leg_length)*(D1[t] - thigh_length + leg_length)*( - D1[t] + thigh_length + leg_length)))**0.5
        lamda = lamda_new
        L1.append(lamda)



    C1 = []
    C2 = []
    #Equation of the point of intersection between two circles imagining that
    for t in np.arange(0, 360,1):
        x1_circs_new = ((x_saddle + F1[t])/2)+(((F1[t] - x_saddle)*(thigh_length**2 - leg_length**2))/(2*D1[t]**2))+(((2*(y_saddle - F2[t]))/(D1[t]**2))*L1[t])
        y1_circs_new = ((y_saddle + F2[t])/2) + (((y_saddle - F2[t])*(thigh_length**2 - leg_length**2))/(2*D1[t]**2)) - (((2*(x_saddle - F1[t]))/(D1[t]**2))*L1[t])
        x1_circs = x1_circs_new
        y1_circs = y1_circs_new
        C1.append(x1_circs)
        C2.append(y1_circs)



    R1 = []
    R2 = []
    #equation of circle with centre at the heel points
    for t in np.arange(0, 360, 1):
        x_leg_new = F1[t] + (leg_length*cos(t*mth.pi/180))
        y_leg_new = F2[t] + (leg_length*sin(t*mth.pi/180))
        x_leg = x_leg_new
        y_leg = y_leg_new
        R1.append(x_leg)
        R2.append(y_leg)



    q_l = np.arange(220,360,10)
    #a general value which 
    #for i in [43, 0, 277]:
        #q = i
        #COSINE FORMULA TO EVALUATE ANGLE, second method
        #P_thigh = ((x_saddle - C1[q])**2 + (y_saddle -C2[q])**2)**0.5
        #P_leg = ((C1[q] - F1[q])**2 + (C2[q] - F2[q])**2)**0.5
        #P_saddle_foot = ((F1[q] - x_saddle)**2 + (F2[q] - y_saddle)**2)**0.5
        #P_torso_knee = ((C1[q] - x_torso)**2 + (C2[q] - y_torso)**2)**0.5
        #P_saddle_torso = ((x_saddle - x_torso)**2 + (y_saddle - y_torso)**2 )**0.5
        #Knee_Angle = 180 - mth.acos(((+ P_thigh**2 + P_leg**2 - P_saddle_foot**2)/(2*P_thigh*P_leg)))*180/mth.pi #kne angle 2
        #Hip_Angle =   mth.acos(((- P_torso_knee**2 + P_leg**2 + P_saddle_torso**2)/(2*P_saddle_torso*P_leg)))*180/mth.pi
        #print(Knee_Angle)
        #print(Hip_Angle)

        #plt.figure(dpi=1000)
        #plt.figure(figsize=(12,8))
        #plt.plot([R1[1], C1[1]], [R2[1], C2[1]])
        #plt.plot([x_saddle, C1[q]], [y_saddle, C2[q]], '-m', linewidth=15) #line going from the 
        #plt.plot(U1, U2)
        #plt.plot(R1, R2)
        #plt.scatter(x_origin, y_origin)
        #plt.plot(S1[110], S2[110])
        #plt.plot([F1[q]+40, C1[q]], [F2[q], C2[q]], '-y', linewidth=10)
        #plt.plot([F1[q], P1[q]], [F2[q], P2[q]], '-g', linewidth=10)
        #plt.plot([x_origin, P1[q]], [y_origin, P2[q]], '-k', linewidth=4) #crank
        #plt.plot([x_saddle, x_torso], [y_saddle, y_torso], '-b', label = 'torso', linewidth=15) #torso
        #plt.scatter(x_saddle, y_saddle)
        #plt.plot(C1, C2) 
        #plt.plot(F1,F2) #heel of foot movement
        #plt.plot(P1,P2) 
        #plt.xlim(-1000, 1400)
        #plt.ylim(-500, 1200)
        #plt.show()

    q=277
    P_thigh = ((x_saddle - C1[277])**2 + (y_saddle -C2[277])**2)**0.5
    P_leg = ((C1[277] - F1[277])**2 + (C2[277] - F2[277])**2)**0.5
    P_saddle_foot = ((F1[277] - x_saddle)**2 + (F2[277] - y_saddle)**2)**0.5
    P_torso_knee = ((C1[277] - x_torso)**2 + (C2[277] - y_torso)**2)**0.5
    P_saddle_torso = ((x_saddle - x_torso)**2 + (y_saddle - y_torso)**2 )**0.5
    Knee_Angle = 180 - mth.acos(((+ P_thigh**2 + P_leg**2 - P_saddle_foot**2)/(2*P_thigh*P_leg)))*180/mth.pi #kne angle 2
    Hip_Angle =   mth.acos(((- P_torso_knee**2 + P_leg**2 + P_saddle_torso**2)/(2*P_saddle_torso*P_leg)))*180/mth.pi

    P_thigh_1 = ((x_saddle - C1[43])**2 + (y_saddle -C2[43])**2)**0.5
    P_leg_1 = ((C1[43] - F1[43])**2 + (C2[43] - F2[43])**2)**0.5
    P_saddle_foot_1 = ((F1[43] - x_saddle)**2 + (F2[43] - y_saddle)**2)**0.5
    P_torso_knee_1 = ((C1[43] - x_torso)**2 + (C2[43] - y_torso)**2)**0.5
    P_saddle_torso_1 = ((x_saddle - x_torso)**2 + (y_saddle - y_torso)**2 )**0.5
    Knee_Angle_1 = 180 - mth.acos(((+ P_thigh_1**2 + P_leg_1**2 - P_saddle_foot_1**2)/(2*P_thigh_1*P_leg_1)))*180/mth.pi #kne angle 2
    Hip_Angle_1 =   mth.acos(((- P_torso_knee_1**2 + P_leg_1**2 + P_saddle_torso_1**2)/(2*P_saddle_torso_1*P_leg_1)))*180/mth.pi
    

    C1_return=int(C1[277])
    C2_return=int(C2[277])
    F1_return=int(int(F1[277])+40)
    F2_return=int(F2[277])
    P1_return=int(P1[277])
    P2_return=int(P2[277])


    #plt.figure(dpi=1000)
    #plt.figure(figsize=(12,8))
    #plt.plot([R1[1], C1[1]], [R2[1], C2[1]])
    #plt.plot([x_saddle, C1[q]], [y_saddle, C2[q]], '-m', linewidth=15) #line going from the 
    #plt.plot(U1, U2)
    #plt.plot(R1, R2)
    #plt.scatter(x_origin, y_origin)
    #plt.plot(S1[110], S2[110])
    #plt.plot([F1[q]+40, C1[q]], [F2[q], C2[q]], '-y', linewidth=10)
    #plt.plot([F1[q], P1[q]], [F2[q], P2[q]], '-g', linewidth=10)
    #plt.plot([x_origin, P1[q]], [y_origin, P2[q]], '-k', linewidth=4) #crank
   # plt.plot([x_saddle, x_torso], [y_saddle, y_torso], '-b', label = 'torso', linewidth=15) #torso
    #plt.scatter(x_saddle, y_saddle)
    #plt.plot(C1, C2) 
    #plt.plot(F1,F2) #heel of foot movement
    #plt.plot(P1,P2) 
    #plt.xlim(-1000, 1400)
    #plt.ylim(-500, 1200)

    SEst = ((x_origin - x_saddle)**2 + (y_origin - y_saddle)**2)**0.5
    saddista = ((C1[q] - x_saddle)**2 + (C2[q] - y_saddle)**2)**0.5


    return Knee_Angle, Hip_Angle,Hip_Angle_1, x_saddle, C1_return, y_saddle, C2_return, F1_return, F2_return, P1_return, P2_return, x_origin, y_origin, x_torso, y_torso, C1, C2, F1, F2, P1, P2 



    















    


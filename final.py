from tkinter import *
from tkinter import ttk
import os
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import numpy as np
from math import sin, cos
from scipy.linalg import eigh
from numpy.linalg import inv

import random

#import routes
from route import route_1_definition, route_2_definition, route_3_definition, route_4_definition, route_5_definition
#import joint angle function
from joint_angle_v2 import joint_angle
#import vibration analysis
from vibrations_final import *

window = Tk()
window.title("Digital Twin")
window.geometry("2500x2500+0+0")


main_frame=Frame(window)
main_frame.pack(fill=BOTH, expand=1)


my_canvas= Canvas(main_frame)
my_canvas.pack(side=LEFT, fill=BOTH, expand=1)


my_scrollbar= ttk.Scrollbar(main_frame, orient=VERTICAL, command=my_canvas.yview)
my_scrollbar.pack(side=RIGHT, fill=Y)

my_x_scrollbar=ttk.Scrollbar(main_frame, orient=HORIZONTAL, command=my_canvas.xview)
my_x_scrollbar.pack(side=BOTTOM, fill=X)


my_canvas.configure(yscrollcommand=my_scrollbar.set)
my_canvas.bind('<Configure>', lambda e: my_canvas.configure(scrollregion=my_canvas.bbox("all")))

my_canvas.configure(xscrollcommand=my_x_scrollbar.set)
my_canvas.bind('<Configure>', lambda e: my_canvas.configure(scrollregion=my_canvas.bbox("all")))

second_frame=Frame(my_canvas)
my_canvas.create_window((0,0),window=second_frame, anchor="nw")
second_frame.rowconfigure([0,1,2,3], minsize=75)
second_frame.columnconfigure([0, 1, 2,3], minsize=100)



frame_numbers= Frame(second_frame, width=70, height=100)
frame_numbers.grid(row=0, column=0)
frame_numbers.rowconfigure([0, 1], minsize=10)
frame_numbers.columnconfigure([0], minsize=100)

frame_input= Frame(frame_numbers, width=70, height=100)
frame_input.grid(row=0, column=0)
frame_input.rowconfigure([0], minsize=10)
frame_input.columnconfigure([0,1,2], minsize=100)


frame_results= Frame(frame_numbers, width=70, height=100)
frame_results.grid(row=1, column=0)
frame_results.rowconfigure([0], minsize=10)
frame_results.columnconfigure([0], minsize=100)

frame_rute= Frame(frame_input, width=70, height=130)
frame_rute.grid(row=0,column=0)
frame_rute.rowconfigure([0, 1, 2,3,4,5,6,7,8,9,10,11,12,13,14], minsize=10)
frame_rute.columnconfigure([0], minsize=100)

frame_rider=Frame(frame_input, width=70, height=130)
frame_rider.grid(row=0, column=1)
frame_rider.rowconfigure([0, 1, 2,3,4,5,6,7,8,9,10,11,12,13], minsize=10)
frame_rider.columnconfigure([0], minsize=100)

frame_bike=Frame(frame_input, width=70, height=130)
frame_bike.grid(row=0, column=2)
frame_bike.rowconfigure([0, 1, 2,3,4,5,6,7,8,9,10,11,12,13], minsize=10)
frame_bike.columnconfigure([0], minsize=100)


frame_elevation=Frame(second_frame, width=400, height=400)
frame_elevation.grid(row=0, column=1)
frame_elevation.rowconfigure([0], minsize=20)
frame_elevation.columnconfigure([0], minsize=100)


frame_speed=Frame(second_frame, width=400, height=400)
frame_speed.grid(row=0, column=2)
frame_speed.rowconfigure([0], minsize=20)
frame_speed.columnconfigure([0], minsize=100)

frame_power=Frame(second_frame, width=400, height=400)
frame_power.grid(row=1, column=0)
frame_power.rowconfigure([0], minsize=20)
frame_power.columnconfigure([0], minsize=100)

frame_power_numbers=Frame(second_frame, width=400, height=400)
frame_power_numbers.grid(row=1, column=1)
frame_power_numbers.rowconfigure([0], minsize=20)
frame_power_numbers.columnconfigure([0], minsize=100)

frame_energy=Frame(second_frame, width=400, height=400)
frame_energy.grid(row=1, column=2)
frame_energy.rowconfigure([0], minsize=20)
frame_energy.columnconfigure([0], minsize=100)

frame_vibration_dis=Frame(second_frame, width=400, height=400)
frame_vibration_dis.grid(row=0, column=3)
frame_vibration_dis.rowconfigure([0], minsize=20)
frame_vibration_dis.columnconfigure([0], minsize=100)


frame_vibration_acc=Frame(second_frame, width=400, height=400)
frame_vibration_acc.grid(row=1, column=3)
frame_vibration_acc.rowconfigure([0], minsize=20)
frame_vibration_acc.columnconfigure([0], minsize=100)


frame_vibration_force=Frame(second_frame, width=400, height=400)
frame_vibration_force.grid(row=2, column=3)
frame_vibration_force.rowconfigure([0], minsize=20)
frame_vibration_force.columnconfigure([0], minsize=100)


                

#rider weight
rider_w = Label(master= frame_rider, text="Rider weight (Kg)")
rider_w_entry=Entry(master=frame_rider)
rider_w.grid(row=0, column=0)
rider_w_entry.grid(row=1, column=0)
rider_w_entry.insert(0, "79")



#rider height
rider_h = Label(master= frame_rider, text="Rider height (cm)")
rider_h_entry=Entry(master= frame_rider)
rider_h.grid(row=2, column=0)
rider_h_entry.grid(row=3, column=0)
rider_h_entry.insert(0,"175")


#power output
power = Label(master= frame_rider, text="Constant power (W)")
power_entry=Entry(master= frame_rider)
power.grid(row=4, column=0)
power_entry.grid(row=5, column=0)
power_entry.insert(0,"150")



#backpack
backpack_experience=Label(master=frame_rider, text="Backpack")
backpack_experience.grid(row=6 ,column=0)

backpack_menu=Menubutton(master=frame_rider, text="         ")
backpack_menu.menu=Menu(backpack_menu)
backpack_menu["menu"]=backpack_menu.menu

No_backpack= IntVar()
backpack=IntVar()

backpack_menu.menu.add_checkbutton(label="No Backapck",variable=No_backpack)
backpack_menu.menu.add_checkbutton(label="Backpack",variable=backpack)

backpack_menu.grid(row=7, column=0)


#mass bike
mass = Label(master= frame_bike,text="Mass of bike (Kg)")
mass_entry=Entry(master= frame_bike)
mass.grid(row=0, column=0)
mass_entry.grid(row=1, column=0)
mass_entry.insert(0,"12.5")



#tyre pressure
tyre_pressure= Label(master= frame_bike, text="Tyre pressure (Bar)")
tyre_pressure_entry=Entry(master= frame_bike)
tyre_pressure.grid(row=2, column=0)
tyre_pressure_entry.grid(row=3, column=0)
tyre_pressure_entry.insert(0,"5.17")

#crank length
crank= Label(master= frame_bike, text="Crank length (mm)")
crank_entry=Entry(master= frame_bike)
crank.grid(row=4, column=0)
crank_entry.grid(row=5, column=0)
crank_entry.insert(0,"170")


#gear ratio
gear=Label(master=frame_bike, text="Gear Ratio")
gear.grid(row=6 ,column=0)

gear_menu=Menubutton(master=frame_bike, text="         ")
gear_menu.menu=Menu(gear_menu)
gear_menu["menu"]=gear_menu.menu

speed_6= IntVar()
speed_3= IntVar()
speed_2= IntVar()
speed_1= IntVar()


gear_menu.menu.add_checkbutton(label="BWR 6-speed",variable=speed_6)
gear_menu.menu.add_checkbutton(label="BSR 3-speed",variable=speed_3)
gear_menu.menu.add_checkbutton(label=" 2-speed",variable=speed_2)
gear_menu.menu.add_checkbutton(label="single speed",variable=speed_1)


gear_menu.grid(row=7, column=0)




#wheel size
wheel= Label(master= frame_bike, text="Wheel size (mm)")
wheel_entry=Entry(master= frame_bike)
wheel.grid(row=8, column=0)
wheel_entry.grid(row=9, column=0)
wheel_entry.insert(0,"406")


#handelbar
handelbar=Label(master=frame_rider, text="Handelbar")
handelbar.grid(row=8 ,column=0)

handelbar_menu=Menubutton(master=frame_rider, text="         ")
handelbar_menu.menu=Menu(handelbar_menu)
handelbar_menu["menu"]=handelbar_menu.menu

s_type= IntVar()
h_type= IntVar()


handelbar_menu.menu.add_checkbutton(label="S-type",variable=s_type)
handelbar_menu.menu.add_checkbutton(label="H-type",variable=h_type)


handelbar_menu.grid(row=9, column=0)



#route
route=Label(master=frame_rute, text="Route selection")
route.grid(row=0 ,column=0)

route_menu=Menubutton(master=frame_rute, text="         ")
route_menu.menu=Menu(route_menu)
route_menu["menu"]=route_menu.menu

Route_1= IntVar()
Route_2= IntVar()
Route_3= IntVar()
Route_4= IntVar()
Route_5= IntVar()

route_menu.menu.add_checkbutton(label="Route 1",variable=Route_1)
route_menu.menu.add_checkbutton(label="Route 2",variable=Route_2)
route_menu.menu.add_checkbutton(label="Route 3",variable=Route_3)
route_menu.menu.add_checkbutton(label="Route 4",variable=Route_4)
route_menu.menu.add_checkbutton(label="Route 5",variable=Route_5)

route_menu.grid(row=1, column=0)

#surface
surface=Label(master=frame_rute, text="Surface")
surface.grid(row=2 ,column=0)

surface_menu=Menubutton(master=frame_rute, text="         ")
surface_menu.menu=Menu(surface_menu)
surface_menu["menu"]=surface_menu.menu

gravel= IntVar()
asphalt= IntVar()
wet= IntVar()

surface_menu.menu.add_checkbutton(label="Asphalt",variable=asphalt)

surface_menu.menu.add_checkbutton(label="Gravel",variable=gravel)
surface_menu.menu.add_checkbutton(label="Wet",variable=wet)

surface_menu.grid(row=3, column=0)



#wind speed
wind_speed = Label(master=frame_rute, text="Wind speed (m/s)")
wind_speed_entry=Entry(master=frame_rute)
wind_speed.grid(row=4, column=0)
wind_speed_entry.grid(row=5, column=0)
wind_speed_entry.insert(0, "0")

#wind direction
wind_direction = Label(master=frame_rute, text="Wind direction")
wind_direction_entry=Entry(master=frame_rute)
wind_direction.grid(row=6,column=0)
wind_direction_entry.grid(row=7, column=0)
wind_direction_entry.insert(0, "0")


#results



results=Label(master= frame_results, text="Results")
results.grid(row=0, column=0)

results_entry= Text(master=frame_results, height=20, width=60)
results_entry.grid(row=1, column =0)






#constant values for the equation
#mass bike kg
m_b = 0
#mass rider kg
m_r = 0
#total mass kg
m= 0
#drag coefficient
cd=0
#air density
o=1.2
#frontal area m
A=0
#wheel diameter
d=0
#wheel radius m
r = 0
#moment of inertia kgm^2
I = 0
#gravity m/s^2
g=9.81
#coefficient of friction
mu=0
#bearing coefficient
b_0=0.091
#bearing coefficeient Ns/m
b_1=0.0087
#mechanical gear ratio
ganma=0
#chain efficiency
n=0.975
#crank length m
lc=0


#get values
def get_values():
    #backpack
    no=No_backpack.get()
    yes=backpack.get()
    backpack_weight=5
    global m_b
    m_b = float( mass_entry.get())
    global m_r
    m_r = float(rider_w_entry.get())
    global d
    d= float( wheel_entry.get())
    global p
    p=float(tyre_pressure_entry.get())
    global lc
    lc=float(crank_entry.get())
    global m

    m= m_b + m_r

    if yes==1:
        m=m+backpack_weight
    
    global r
    r=float((d/2)/1000)









#route selection and outputting route profile    
def get_profile():
    route_1=Route_1.get()
    route_2=Route_2.get()
    route_3=Route_3.get()
    route_4=Route_4.get()
    route_5=Route_5.get()
    if route_1==1:
        gradient, elevation, distance, dis=route_1_definition()
    
    elif route_2==1:
        gradient, elevation, distance, dis=route_2_definition()

    elif route_3==1:
        gradient, elevation, distance, dis=route_3_definition()
    
    elif route_4==1:
        gradient, elevation, distance, dis=route_4_definition()

    elif route_5==1:
        gradient, elevation, distance, dis=route_5_definition()
        
    return gradient, elevation, distance, dis


#cadence
def calculate_cadence(speed):
    wheel_diameter=float( wheel_entry.get())
    wheel_radius=wheel_diameter/2
    crank=float(crank_entry.get())
    s_6=speed_6.get()
    s_3=speed_3.get()
    s_2=speed_2.get()
    s_1=speed_1.get()
    recomended_cadence=80
    cadence=[]
    gear_ratio_list=[]
    ideal_gear_ratio=[]
    length=len(speed)
    if s_6==1:
        gear_ratio_1=float(2.63)
        gear_ratio_2=float(3.24)
        gear_ratio_3=float(4.23)
        gear_ratio_4=float(5.08)
        gear_ratio_5=float(6.47)
        gear_ratio_6=float(7.96)
        for i in range(length):
            v=speed[i]
            ideal_gear=(v*60)/((wheel_radius/1000)*recomended_cadence*2*(math.pi))
            ideal_gear_ratio.append(ideal_gear)
            cadence_1=(v*60)/((wheel_radius/1000)*gear_ratio_1*2*(math.pi))
            dif_1= abs(cadence_1-recomended_cadence)
            cadence_2=(v*60)/((wheel_radius/1000)*gear_ratio_2*2*(math.pi))
            dif_2= abs(cadence_2-recomended_cadence)
            cadence_3=(v*60)/((wheel_radius/1000)*gear_ratio_3*2*(math.pi))
            dif_3= abs(cadence_3-recomended_cadence)
            cadence_4=(v*60)/((wheel_radius/1000)*gear_ratio_4*2*(math.pi))
            dif_4= abs(cadence_4-recomended_cadence)
            cadence_5=(v*60)/((wheel_radius/1000)*gear_ratio_5*2*(math.pi))
            dif_5= abs(cadence_5-recomended_cadence)
            cadence_6=(v*60)/((wheel_radius/1000)*gear_ratio_6*2*(math.pi))
            dif_6= abs(cadence_6-recomended_cadence)
            diference=[dif_1, dif_2, dif_3,dif_4, dif_5, dif_6]
            if min(diference)==dif_1:
                cadence.append(cadence_1)
                gear_ratio_list.append(gear_ratio_1)
            elif min(diference)==dif_2:
                cadence.append(cadence_2)
                gear_ratio_list.append(gear_ratio_2)
            elif min(diference)==dif_3:
                cadence.append(cadence_3)
                gear_ratio_list.append(gear_ratio_3)
            elif min(diference)==dif_4:
                cadence.append(cadence_4)
                gear_ratio_list.append(gear_ratio_4)
            elif min(diference)==dif_5:
                cadence.append(cadence_5)
                gear_ratio_list.append(gear_ratio_5)
            elif min(diference)==dif_6:
                cadence.append(cadence_6)
                gear_ratio_list.append(gear_ratio_6)

        
    elif s_3==1:
        gear_ratio_1=float(3.54)
        gear_ratio_2=float(4.72)
        gear_ratio_3=float(6.29)
        for i in range(length):
            v=speed[i]
            ideal_gear=(v*60)/((wheel_radius/1000)*recomended_cadence*2*(math.pi))
            ideal_gear_ratio.append(ideal_gear)
            cadence_1=(v*60)/((wheel_radius/1000)*gear_ratio_1*2*(math.pi))
            dif_1= abs(cadence_1-recomended_cadence)
            cadence_2=(v*60)/((wheel_radius/1000)*gear_ratio_2*2*(math.pi))
            dif_2= abs(cadence_2-recomended_cadence)
            cadence_3=(v*60)/((wheel_radius/1000)*gear_ratio_3*2*(math.pi))
            dif_3= abs(cadence_3-recomended_cadence)
            diference=[dif_1, dif_2, dif_3]
            if min(diference)==dif_1:
                cadence.append(cadence_1)
                gear_ratio_list.append(gear_ratio_1)
            elif min(diference)==dif_2:
                cadence.append(cadence_2)
                gear_ratio_list.append(gear_ratio_2)
            elif min(diference)==dif_3:
                cadence.append(cadence_3)
                gear_ratio_list.append(gear_ratio_3)
    elif s_2==1:
        gear_ratio_1=float(4.46)
        gear_ratio_2=float(5.95)
        for i in range(length):
            v=speed[i]
            ideal_gear=(v*60)/((wheel_radius/1000)*recomended_cadence*2*(math.pi))
            ideal_gear_ratio.append(ideal_gear)
            cadence_1=(v*60)/((wheel_radius/1000)*gear_ratio_1*2*(math.pi))
            dif_1= abs(cadence_1-recomended_cadence)
            cadence_2=(v*60)/((wheel_radius/1000)*gear_ratio_2*2*(math.pi))
            dif_2= abs(cadence_2-recomended_cadence)
            diference=[dif_1, dif_2, ]
            if min(diference)==dif_1:
                cadence.append(cadence_1)
                gear_ratio_list.append(gear_ratio_1)
            elif min(diference)==dif_2:
                cadence.append(cadence_2)
                gear_ratio_list.append(gear_ratio_2)
    elif s_1==1:
        gear_ratio_1=float(5.94)
        for i in range(length):
            v=speed[i]
            ideal_gear=(v*60)/((wheel_radius/1000)*recomended_cadence*2*(math.pi))
            ideal_gear_ratio.append(ideal_gear)
            cadence_1=(v*60)/((wheel_radius/1000)*gear_ratio_1*2*(math.pi))
            cadence.append(cadence_1)
            gear_ratio_list.append(gear_ratio_1)
        
        
                   
    return cadence, gear_ratio_list, ideal_gear_ratio


def number_of_gear_changes(gear_ratio_list):
    length=len(gear_ratio_list)
    number_changes=0
    i=1
    j=0
    while i< length:
        gear=gear_ratio_list[j]
        gear_1=gear_ratio_list[i]
        i=i+1
        j=j+1
        if gear != gear_1:
            number_changes=number_changes+1
    return number_changes 
        
    

#force exerted on the pedal 
def calculate_force(power, speed, gear_ratio_list):
    force=[]
    wheel_diameter=float( wheel_entry.get())
    wheel_radius=float((wheel_diameter/2)/1000)
    crank=float(float(crank_entry.get())/1000)
    
    circumference=2*(math.pi)*wheel_radius
    length=len(speed)
    for i in range(length):
        P=power[i]
        v=speed[i]
        gear_ratio=gear_ratio_list[i]
        F=float((P*wheel_radius*gear_ratio)/(v*crank))
        force.append(F)
    return force


#defining mu based on inputs 
def get_terrain():
    Asphalt=asphalt.get()
    Gravel=gravel.get()
    Wet=wet.get()
    P_tyre=float(tyre_pressure_entry.get())
    P_tyre_kpa=float(P_tyre*100)
    
    if Asphalt==1:
        global mu
        mu=float(0.1071*(P_tyre_kpa**(-0.477)))
    
    elif Gravel==1:
        mu=float(0.007)

    elif Wet==1:
        mu=0.003
    return mu


#areodynamics
def coefficient_drag(v,gra):
    height=float(rider_h_entry.get())
    h= float(height/100)
    #handelbar
    type_s= s_type.get()
    type_h= h_type.get()
    #backpack
    no=No_backpack.get()
    yes=backpack.get()
    #no backpack
    if gra<0.08 and no==1:
        if type_s==1:
            if  h<=1.70:
                cd=  1.413 - 0.024*v
            elif h>1.70 and h<1.80:
                cd=  1.401 - 0.021*v
            elif h>=1.80:
                 cd=  1.386 - 0.018*v
         
        elif type_h==1:
            if  h<=1.70:
                cd= 1.444 - 0.021*v
            elif h>1.70 and h<1.80:
                cd= 1.403 - 0.014*v
            elif h>=1.80:
                cd=  1.473 - 0.019*v
        else:
            print ("No handlebar or backcpack selected")
    
    elif gra>=0.08 and no==1:
        if type_s==1:
            if  h<=1.70:
                cd=  1.3856 - 0.0166*v
            elif h>1.70 and h<1.80:
                cd=  1.4369 - 0.0195*v
            elif h>=1.80:
                 cd=  1.4629 - 0.0186*v
         
        elif type_h==1:
            if  h<=1.70:
                cd= 1.4173 - 0.0189*v
            elif h>1.70 and h<1.80:
                cd= 1.4626 - 0.0198*v
            elif h>=1.80:
                cd=  1.4383 - 0.0196*v
        else:
            print ("No handelbar or backcpack selected")
    #with backpack
    if gra<0.08 and yes==1:
        if type_s==1:
            if  h<=1.70:
                cd=  1.3566 - 0.0206*v
            elif h>1.70 and h<1.80:
                cd=  1.3849 - 0.0183*v
            elif h>=1.80:
                 cd=  1.3171 - 0.0168*v
         
        elif type_h==1:
            if  h<=1.70:
                cd= 1.3885 - 0.0120*v
            elif h>1.70 and h<1.80:
                cd= 1.3882 - 0.0158*v
            elif h>=1.80:
                cd=  1.4726 - 0.0185*v
        else:
            print ("No handelbar or backcpack selected")
    
    elif gra>=0.08 and yes==1:
        if type_s==1:
            if  h<=1.70:
                cd=  1.387 - 0.0156*v
            elif h>1.70 and h<1.80:
                cd=  1.4283 - 0.0164*v
            elif h>=1.80:
                 cd=  1.4629 - 0.0186*v
         
        elif type_h==1:
            if  h<=1.70:
                cd= 1.387 - 0.0156*v
            elif h>1.70 and h<1.80:
                cd= 1.4283 - 0.0164*v
            elif h>=1.80:
                cd=  1.4383 - 0.0196*v
        else:
            print ("No handelbar or backcpack selected")

    return cd

    

#defining frontal area and the lambda value based on the ratio of frontal to side and the direction of travel 
def frontal_area(gra, beta ):
    #size
    weight=float(rider_w_entry.get())
    height=float(rider_h_entry.get())
    h= float(height/100)
    
    #handelbar
    type_s= s_type.get()
    type_h= h_type.get()
    #backpack
    yes=backpack.get()
    no=No_backpack.get()
    #calculations
    #no backpack
    if gra<0.08 and no==1:
        if type_s==1:
            A = float(0.5061*h - 0.3545)
            ratio=1.73247399641141
            lam=(cos(beta)**2)+(ratio*(sin(beta)**2))
        elif type_h==1:
            A= float (0.7526*h - 0.753)
            ratio=1.55468077585286
            lam=(cos(beta)**2)+(ratio*(sin(beta)**2))
    elif gra>=0.08 and no==1:
        if type_s==1:
            A = float(0.4653*h - 0.23935)
            ratio=1.49171842310355
            lam=(cos(beta)**2)+(ratio*(sin(beta)**2))
        elif type_h==1:
            A= float (0.431*h - 0.1791)
            ratio=1.42492634014957
            lam=(cos(beta)**2)+(ratio*(sin(beta)**2))
    #with backpack
    if gra<0.08 and yes==1:
        if type_s==1:
            A = float(0.5061*h - 0.3545)
            ratio=1.99111457016106
            lam=(cos(beta)**2)+(ratio*(sin(beta)**2))
        elif type_h==1:
            A= float (0.7526*h - 0.753)
            ratio=1.90088938986641
            lam=(cos(beta)**2)+(ratio*(sin(beta)**2))
    elif gra>=0.08 and yes==1:
        if type_s==1:
            A = float(0.4653*h - 0.23935)
            ratio=1.78017947057954
            lam=(cos(beta)**2)+(ratio*(sin(beta)**2))
        elif type_h==1:
            A= float (0.431*h - 0.1791)
            ratio=1.64311351439815
            lam=(cos(beta)**2)+(ratio*(sin(beta)**2))
        
    return A, lam



def average_speed(speed):
    return sum(speed)/len(speed)

def moment():
    wheel_diameter=float( wheel_entry.get())
    radius=float((wheel_diameter/2)/1000)
    wheel_weight=1
    return 2*wheel_weight*radius**2


def calculate_stress(force, gradient):
    stress=[]
    safety_factor=[]
    for i in range(len(force)):
        if gradient[i]<0.08:
            stress_1=float((0.27*(force[i]))+(1.17*m_r))
            safety=float(618/stress_1)
            stress.append(stress_1)
            safety_factor.append(safety)
        elif gradient[i]>=0.08:
            stress_1=float((0.399* (force[i])) - 3.65)
            safety=float(618/stress_1)
            stress.append(stress_1)
            safety_factor.append(safety)
    max_stress=max(stress)
    min_safety=min(safety_factor)
    return max_stress,min_safety
    


def get_joint_angle():
    height=float(rider_h_entry.get())*10
    crank=float(crank_entry.get())
    Knee_Angle, Hip_Angle, Hip_angle_1, x_saddle, C1_return, y_saddle, C2_return, F1_return, F2_return, P1_return, P2_return, x_origin, y_origin, x_torso, y_torso, C1, C2, F1, F2, P1, P2 =joint_angle(height, crank)

    return Knee_Angle, Hip_Angle, Hip_angle_1, x_saddle, C1_return, y_saddle, C2_return, F1_return, F2_return, P1_return, P2_return, x_origin, y_origin, x_torso, y_torso, C1, C2, F1, F2, P1, P2 

def road_grade():
    Asphalt=asphalt.get()
    Gravel=gravel.get()
    Wet=wet.get()
    if Asphalt==1:
        grade=0
    elif Gravel==1:
        grade=2
    elif Wet==1:
        grade=0
    
    return grade
        

#various plots 
def plot_elevation(distance, elevation):
    #plot elevation
    fig_elevation = plt.figure(figsize=(7, 6), dpi=60)
    ax_elevation = fig_elevation.add_subplot(111)
    ax_elevation.plot(distance, elevation)                
    ax_elevation.set_xlabel("Distance(m)")
    ax_elevation.set_ylabel("Elevation(m)")
    ax_elevation.set_title("Elevation profile")
    ax_elevation.set_ylim(500,1000)
    canvas_elevation = FigureCanvasTkAgg(fig_elevation, master=frame_elevation)
    canvas_elevation.draw()
    canvas_elevation.get_tk_widget().pack()

def plot_power(distance, power):
    #plot power
    fig_power = plt.figure(figsize=(7,6), dpi=60)
    ax_power = fig_power.add_subplot(111)
    ax_power.plot(distance, power)                
    ax_power.set_xlabel("Distance(m)")
    ax_power.set_ylabel("Power(W)")
    ax_power.set_title("Power")
    canvas_power = FigureCanvasTkAgg(fig_power, master=frame_power)
    canvas_power.draw()
    canvas_power.get_tk_widget().pack()
    

def plot_energy(distance, energy):
    fig_energy = plt.figure(figsize=(7, 6), dpi=60)
    ax_energy = fig_energy.add_subplot(111)
    ax_energy.plot(distance, energy)                
    ax_energy.set_xlabel("Distance(m)")
    ax_energy.set_ylabel("Energy(J)")
    ax_energy.set_title("Energy")
    canvas_energy = FigureCanvasTkAgg(fig_energy, master=frame_energy)
    canvas_energy.draw()
    canvas_energy.get_tk_widget().pack()


def plot_speed(distance, speed):
    fig_speed = plt.figure(figsize=(7,6), dpi=60)
    ax_speed = fig_speed.add_subplot(111)
    ax_speed.plot(distance, speed)                
    ax_speed.set_xlabel("Distance(m)")
    ax_speed.set_ylabel("Speed(m/s)")
    ax_speed.set_title("Speed")
    canvas_speed = FigureCanvasTkAgg(fig_speed, master=frame_speed)
    canvas_speed.draw()
    canvas_speed.get_tk_widget().pack()




def plot_power_numbers(distance, Power_d, Power_p, Power_r, Power_b, Power_k):
    fig_power_numbers = plt.figure(figsize=(7,6), dpi=60)
    ax_power_numbers = fig_power_numbers.add_subplot(111)
    ax_power_numbers.plot(distance, Power_d, label="Drag")
    ax_power_numbers.plot(distance, Power_p, label="Potential")
    ax_power_numbers.plot(distance, Power_r, label="Rolling")
    ax_power_numbers.plot(distance, Power_b, label="Bearing")
    ax_power_numbers.plot(distance, Power_k, label="Kinetic")
    ax_power_numbers.set_xlabel("Distance(m)")
    ax_power_numbers.set_ylabel("Power (W)")
    ax_power_numbers.set_title("Power")
    ax_power_numbers.legend()
    canvas_power_numbers = FigureCanvasTkAgg(fig_power_numbers, master=frame_power_numbers)
    canvas_power_numbers.draw()
    canvas_power_numbers.get_tk_widget().pack()


def plot_force(distance, force):
    fig_force = plt.figure(figsize=(7,6), dpi=60)
    ax_force = fig_force.add_subplot(111)
    ax_force.plot(distance, force)                
    ax_force.set_xlabel("Distance(m)")
    ax_force.set_ylabel("Force(N)")
    ax_force.set_title("Force exerted on the pedals")
    canvas_force = FigureCanvasTkAgg(fig_force, master=frame_speed)
    canvas_force.draw()
    canvas_force.get_tk_widget().pack()



def plot_cadence(distance, cadence):
    fig_cadence = plt.figure(figsize=(7,6), dpi=60)
    ax_cadence = fig_cadence.add_subplot(111)
    ax_cadence.plot(distance, cadence)                
    ax_cadence.set_xlabel("Distance(m)")
    ax_cadence.set_ylabel("Cadence (rpm)")
    ax_cadence.set_title("Cadence")
    canvas_cadence = FigureCanvasTkAgg(fig_cadence, master=frame_elevation)
    canvas_cadence.draw()
    canvas_cadence.get_tk_widget().pack()


def plot_acc(distance, acceleration):
    fig_acceleration = plt.figure(figsize=(7,6), dpi=60)
    ax_acceleration = fig_acceleration.add_subplot(111)
    ax_acceleration.plot(distance, acceleration)
    ax_acceleration.set_xlabel("Distance(m)")
    ax_acceleration.set_ylabel("Acceleration(m/s^2)")
    ax_acceleration.set_title("Acceleration")
    canvas_acceleration = FigureCanvasTkAgg(fig_acceleration, master=frame_power_numbers)
    canvas_acceleration.draw()
    canvas_acceleration.get_tk_widget().pack()


def plot_gear(distance, gear):
    fig_gear = plt.figure(figsize=(7,6), dpi=60)
    ax_gear = fig_gear.add_subplot(111)
    ax_gear.plot(distance, gear)
    ax_gear.set_xlabel("Distance(m)")
    ax_gear.set_ylabel("Gear Ratio")
    ax_gear.set_title("Recomended gear ratio")
    canvas_gear = FigureCanvasTkAgg(fig_gear, master=frame_power_numbers)
    canvas_gear.draw()
    canvas_gear.get_tk_widget().pack()


def plot_psd(F2, problem, problem2):
    fig_psd = plt.figure(figsize=(7,6), dpi=60)
    ax_psd = fig_psd.add_subplot(111)
    ax_psd.plot(F2, problem )
    ax_psd.plot(F2, problem2)
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("PSD")
    ax_psd.set_title("PSD")
    ax_psd.set_xlim(0,80)
    canvas_psd = FigureCanvasTkAgg(fig_psd, master=frame_vibration_dis)
    canvas_psd.draw()
    canvas_psd.get_tk_widget().pack()

def plot_vibrations(time, G6, G7):
    fig_vibrations = plt.figure(figsize=(7,6), dpi=60)
    ax_vibrations= fig_vibrations.add_subplot(111)
    ax_vibrations.plot(time, G6 , label="Seat")
    ax_vibrations.plot(time, G7, label="Handelbar")
    ax_vibrations.set_xlabel("Time(s)")
    ax_vibrations.set_ylabel("Acceleration (m/^2)")
    ax_vibrations.set_title(" Vibrations Acceleration response")
    ax_vibrations.set_xlim(0,1)
    ax_vibrations.legend()
    canvas_vibrations = FigureCanvasTkAgg(fig_vibrations, master=frame_vibration_dis)
    canvas_vibrations.draw()
    canvas_vibrations.get_tk_widget().pack()

def plot_joint():
    Knee_angle, Hip_angle,Hip_angle_1, x_saddle, C1_return, y_saddle, C2_return, F1_return, F2_return, P1_return, P2_return, x_origin, y_origin, x_torso, y_torso, C1, C2, F1, F2, P1, P2 =get_joint_angle()
    fig_joint = plt.figure(figsize=(7,6), dpi=60)
    ax_joint = fig_joint.add_subplot(111)
    ax_joint.plot([x_saddle, C1_return], [y_saddle, C2_return], '-m', linewidth=15)
    ax_joint.scatter(x_origin, y_origin)
    ax_joint.plot([F1_return, C1_return], [F2_return, C2_return], '-y', linewidth=10)
    ax_joint.plot([F1_return, P1_return], [F2_return, P2_return], '-g', linewidth=10)
    ax_joint.plot([x_origin, P1_return], [y_origin, P2_return], '-k', linewidth=4) #crank
    ax_joint.plot([x_saddle, x_torso], [y_saddle, y_torso], '-b', label = 'torso', linewidth=15) #torso
    ax_joint.scatter(x_saddle, y_saddle)
    ax_joint.plot(C1, C2) 
    ax_joint.plot(F1,F2) #heel of foot movement
    ax_joint.plot(P1,P2) 
    ax_joint.set_xlim(-1000, 1400)
    ax_joint.set_ylim(-500, 1200)
    ax_joint.set_title("Joint Angles")
    canvas_joint = FigureCanvasTkAgg(fig_joint, master=frame_vibration_acc)
    canvas_joint.draw()
    canvas_joint.get_tk_widget().pack()
    
    


def calculate_power_constant_speed():
    values=get_values()
    velocity=float(speed_entry.get())
    total_time=float(dis/velocity)
    gradient, elevation, distance, dis=get_profile()
    mu = get_terrain()
    speed=[]
    speed_1=velocity
    for i in range(0, dis):
        speed.append(speed_1)
    a=0
    Power=[]
    Power_d=[]
    Power_p=[]
    Power_r=[]
    Power_b=[]
    Power_k=[]
    energy=0
    energy_1=[]
    s=float(1/velocity)#time interval
    for q in range(0, dis):
        v=speed[q]
        gra=gradient[q]
        P_d=float( 0.5*A*o*cd*(v**3))
        Power_d.append(P_d)
        P_p=float(m*g*v*gra)
        Power_p.append(P_p)
        P_r=float(m*g*v*mu)
        Power_r.append(P_r)
        P_b=float((beta+ beta_1*v)*v)
        Power_b.append(P_b)
        P_k=float((m+(I/r))*a*v)
        Power_k.append(P_k)
        P=float((P_d+P_p+P_r+P_b+P_k))
        Power.append(P)
        if P>=0:
            energy_2=float(P*s)
            energy=energy+energy_2
            energy_1.append(energy)
        else:
            energy=energy
            energy_1.append(energy)
    kcal=float(energy/4184)
    print (kcal)
    force = calculate_force(Power, speed)
    cadence=calculate_cadence(speed)
    #plot elevation
    plot_elevation(distance,elevation)
    #plot power
    plot_power(distance, Power)
    #plot energy
    plot_energy(distance,energy_1)
    #plot speed
    plot_speed(distance, speed)
    #plot power numbers
    plot_power_numbers(distance, Power_d, Power_p, Power_r, Power_b, Power_k)
    #plot force
    plot_force(distance, force)
    #plot cadence
    plot_cadence(distance, cadence)

    return distance, elevation, Power, energy_1, speed, Power_d, Power_p, Power_r, Power_b, Power_k
    




def calculate_speed_constant_power():
    values=get_values()
    gradient, elevation, distance, dis=get_profile()
    mu = get_terrain()
    time_step=1
    P=float(power_entry.get())
    Power=[]
    distance_covered=int(1)
    distance_1=[]
    speed=[]
    acceleration=[]
    total_time=0
    time=[]
    energy=0
    energy_1=[]

    time_standing=0
    I=moment()

    gradient_2=[]

    i=0
    v=1
    while distance_covered<= dis:
        i=i+1
        total_time=total_time+time_step
        time.append(total_time)

        index=distance.index(distance_covered)
        gra=gradient[index]
        gradient_2.append(gra)

        if gra>=0.08:
            time_standing=time_standing+ time_step
            
        #wind values
        b_o=math.radians(float(wind_direction_entry.get()))
        v_a=float(wind_speed_entry.get())
    
        v_w=math.sqrt(((v_a*sin(b_o))**2)+((v+v_a*cos(b_o))**2))
        b_a=math.acos((v+v_a*cos(b_o))/v_w)
        
        A, lam =frontal_area(gra, b_a)
        cd=coefficient_drag(v_w,gra)

        acc=float((n*P-m*g*gra*v-0.5*lam*A*o*cd*((v_w)**2)*v*cos(b_a)-mu*m*g*v-(b_0+b_1*v))/((m+(I/r**2))*v*0.5))
        acceleration.append(acc)
        v=v+time_step*acc
        speed.append(v)
        distance_covered=round(distance_covered+time_step*v)
        distance_1.append(distance_covered)
        Power.append(P)
        if P>=0:
            energy_2=float(P*time_step)
            energy=energy+energy_2
            energy_1.append(energy)
        else:
            energy=energy
            energy_1.append(energy)
    kcal=float(energy/4184)
    #print ("Energy (kcal)=", kcal)
    cadence, gear_ratio_list, ideal_gear_ratio=calculate_cadence(speed)
    force = calculate_force(Power, speed, gear_ratio_list)

    average_v=average_speed(speed)
   # print("Average speed (m/s)=", average_v)
    
    #print("total riding time(s)=" , total_time)
    #print ("time standing up(s)=", time_standing)

    per_standing=(time_standing*100)/total_time

    #print("percentage standing=", per_standing)

    number_changes =number_of_gear_changes(gear_ratio_list)
    #print ("number of gear changes =", number_changes)

    max_stress, min_safety=calculate_stress(force, gradient_2)
    #print ("max stress=", max_stress)
    #print ("min safety factor=",min_safety)

    return gradient, elevation, distance, acceleration, Power, distance_1,energy_1, speed, force, cadence, gear_ratio_list, ideal_gear_ratio, average_v, kcal, total_time, time_standing, number_changes, max_stress, min_safety


def plot_gui():
    gradient, elevation, distance, acceleration, Power, distance_1, energy_1, speed, force, cadence, gear_ratio_list, ideal_gear_ratio, average_v, kcal,total_time, time_standing, number_changes, max_stress, min_safety=calculate_speed_constant_power()
    #plot elevation
    plot_elevation(distance,elevation)
    #plot power
    plot_power(distance_1, Power)
    #plot energy
    plot_energy(distance_1,energy_1)
    #plot speed
    plot_speed(distance_1, speed)
    #plot force
    plot_force(distance_1, force)
    #plot cadence
    plot_cadence(distance_1, cadence)
    #plot acceleration
    #plot_acc(distance_1, acceleration)
    #plot gear ratio
    plot_gear(distance_1, gear_ratio_list)

    plot_joint()

    grade= road_grade()

    F2, problem, problem2, time, G6, G7, A1_rms, A2_rms, Z_rms=vibrations(average_v, m_r, grade)

    plot_psd(F2, problem, problem2)
    
    plot_vibrations(time, G6, G7)

    Knee_angle, Hip_angle, Hip_angle_1, x_saddle, C1_return, y_saddle, C2_return, F1_return, F2_return, P1_return, P2_return, x_origin, y_origin, x_torso, y_torso, C1, C2, F1, F2, P1, P2 =get_joint_angle()
    #print ("Knee angle=",Knee_angle)
    #print("Hip angle=",Hip_angle)

    Knee_text = "Knee angle at BDC (deg)= {} \n".format(Knee_angle)
    Hip_text = "Hip angle at TDC (deg)= {} \n".format(Hip_angle_1)
    Energy_text="Energy (kcal)={}\n".format(kcal)
    average_v_text="Average velocity (m/s)={}\n".format(average_v)
    total_time_text ="Total time (s)={}\n".format(total_time)
    time_standing_text="Time standing (s)={}\n".format(time_standing)
    number_changes_text="Number of gear changes={}\n".format(number_changes)
    max_stress_text="Max stress(MPa)={}\n".format(max_stress)
    min_safety_text="Min safety factor={}\n".format(min_safety)
    A1_rms_text=" A1 RMS ={}\n".format(A1_rms)
    A2_rms_text=" A2 RMS ={}\n".format(A2_rms)
    Z_rms_text=" Z RMS ={}\n".format(Z_rms)
    
    results_entry.insert(END, total_time_text)
    results_entry.insert(END, Energy_text)
    results_entry.insert(END, average_v_text)
    results_entry.insert(END, time_standing_text)
    results_entry.insert(END, number_changes_text)
    results_entry.insert(END, max_stress_text)
    results_entry.insert(END, min_safety_text)
    results_entry.insert(END, Knee_text)
    results_entry.insert(END, Hip_text)
    results_entry.insert(END, A1_rms_text)
    results_entry.insert(END, A2_rms_text)
    results_entry.insert(END, Z_rms_text)

    



def plot():
    gradient, elevation, distance, acceleration, Power, distance_1, energy_1, speed, force, cadence, gear_ratio_list, ideal_gear_ratio, average_v, kcal,total_time, time_standing, number_changes, max_stress, min_safety=calculate_speed_constant_power()
    

    plt.plot(distance, elevation)
    plt.xlabel("Distance(m)")
    plt.ylabel("Elevation(m)")
    plt.title("Elevation profile")
    plt.show()

    plt.plot(distance_1, speed)
    plt.xlabel("Distance(m)")
    plt.ylabel("Speed(m/s)")
    plt.title("Speed")
    plt.show()

    plt.plot(distance_1, force)
    plt.xlabel("Sistance(m)")
    plt.ylabel("Force(N)")
    plt.title("Force")
    plt.show()

    plt.plot(distance_1, cadence)
    plt.xlabel("Distance(m)")
    plt.ylabel("Cadence (rpm)")
    plt.title("Cadence")
    plt.show()


    plt.plot(distance_1, energy_1)
    plt.xlabel("Distance(m)")
    plt.ylabel("Energy (J)")
    plt.title("Energy Expenditure")
    plt.show()


    plt.plot(distance_1, gear_ratio_list)
    plt.xlabel("Distance(m)")
    plt.ylabel("Gear ratio")
    plt.title("Recomended gear ratio")
    plt.show()


    plt.plot(distance_1, ideal_gear_ratio)
    plt.xlabel("Distance(m)")
    plt.ylabel("Gear ratio")
    plt.title("Recomended gear ratio")
    plt.show()


    grade= road_grade()



    F2, problem, problem2, time, G6, G7, A1_rms, A2_rms, Z_rms=vibrations(average_v, m, grade)

    
    plt.plot(F2, problem, color = 'g')
    plt.plot(F2, problem2, color= 'b')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.xlim(0,80)
    plt.title('PSD ')
    plt.show()


    plt.plot(time, G6)
    plt.plot(time, G7)
    plt.xlim(0, 1)
    plt.title(' Vibrations Acceleration Response (m/s^2)')
    plt.xlabel('time (s)')
    plt.ylabel('Acceleration (ms-2)')
    plt.legend(['A1','A2','A3','A4','A5','A6','A7'], loc='lower right')
    plt.show()


    Knee_angle, Hip_angle, Hip_angle_1, x_saddle, C1_return, y_saddle, C2_return, F1_return, F2_return, P1_return, P2_return, x_origin, y_origin, x_torso, y_torso, C1, C2, F1, F2, P1, P2 =get_joint_angle()

    #plt.figure(dpi=1000)
    #plt.figure(figsize=(12,8))
    plt.plot([x_saddle, C1_return], [y_saddle, C2_return], '-m', linewidth=15) #line going from the 
    plt.scatter(x_origin, y_origin)
    plt.plot([F1_return, C1_return], [F2_return, C2_return], '-y', linewidth=10)
    plt.plot([F1_return, P1_return], [F2_return, P2_return], '-g', linewidth=10)
    plt.plot([x_origin, P1_return], [y_origin, P2_return], '-k', linewidth=4) #crank
    plt.plot([x_saddle, x_torso], [y_saddle, y_torso], '-b', label = 'torso', linewidth=15) #torso
    plt.scatter(x_saddle, y_saddle)
    plt.plot(C1, C2) 
    plt.plot(F1,F2) #heel of foot movement
    plt.plot(P1,P2) 
    plt.xlim(-1000, 1400)
    plt.ylim(-500, 1200)
    plt.show()

    #print ("Knee angle=",Knee_angle)
    #print("Hip angle=",Hip_angle)

    Knee_text = "Knee angle at BDC (deg)= {} \n".format(Knee_angle)
    Hip_text = "Hip angle  at TDC (deg)= {} \n".format(Hip_angle_1)
    Energy_text="Energy (kcal)={}\n".format(kcal)
    average_v_text="Average velocity (m/s)={}\n".format(average_v)
    total_time_text ="Total time (s)={}\n".format(total_time)
    time_standing_text="Time standing (s)={}\n".format(time_standing)
    number_changes_text="Number of gear changes={}\n".format(number_changes)
    max_stress_text="Max stress(MPa)={}\n".format(max_stress)
    min_safety_text="Min safety factor={}\n".format(min_safety)
    A1_rms_text=" A1 RMS ={}\n".format(A1_rms)
    A2_rms_text=" A2 RMS ={}\n".format(A2_rms)
    Z_rms_text=" Z RMS ={}\n".format(Z_rms)
    
    results_entry.insert(END, total_time_text)
    results_entry.insert(END, Energy_text)
    results_entry.insert(END, average_v_text)
    results_entry.insert(END, time_standing_text)
    results_entry.insert(END, number_changes_text)
    results_entry.insert(END, max_stress_text)
    results_entry.insert(END, min_safety_text)                                                   
    results_entry.insert(END, Knee_text)
    results_entry.insert(END, Hip_text)
    results_entry.insert(END, A1_rms_text)
    results_entry.insert(END, A2_rms_text)
    results_entry.insert(END, Z_rms_text)









#run constant power
run_btn_power=Button( master= frame_rute, text="Run constant power",
                         command=plot, width=15, height=2)
run_btn_power.grid(row=8, column=0)


#plot graph to be saved
plot_graphs=Button(master=frame_rute, text="Plot graphs",
                   command=plot_gui, width=15, height=2)
plot_graphs.grid(row=9, column=0)




window.mainloop()



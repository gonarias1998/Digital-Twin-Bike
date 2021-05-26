import numpy as np
from math import sin
from scipy.linalg import eigh
from numpy.linalg import inv
from matplotlib import pyplot as plt
import random
from scipy.signal import butter,filtfilt

def butter_lowpass_filter(Z1, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, Z1)
    return y
    
# def F(t, i):
#     F = np.zeros((2*dof,1))
#     F[2] = k4*A1[i] * np.sin((O1[i]*t*velocity) + T1[i])
#     return F

def F(t, i):
    F = np.zeros((2*dof,1))
    F[2] = filteredforce[i]
    return F


def P(Y,t,i):
    return A_inv.dot( F(t,i) - B.dot(Y) )

def RK4_step(Y, t, time_step,i):
	J1 = P(Y,t,i)
	J2 = P(Y+0.5*J1*time_step, t+0.5*time_step,i)
	J3 = P(Y+0.5*J2*time_step, t+0.5*time_step,i)
	J4 = P(Y+J3*time_step, t+time_step,i)
	return time_step * (J1 + 2*J2 + 2*J3 + J4) /6

    


# setup the parameters
omega = 10 #forcing frequency
velocity = 6.53     #6.1 #4.73 #6.53 1st route #speed of bike in meters per second
#springs
k1 = 200000 #Rear Tire stiffness ############ Hard Tire-rim-high pressure
k2 = 170500 # Rear triangle 
k3 = 200000 # front tire stiffness  ########### Hard tire-rim-high pressure
k4 = 90000# Fork Stiffness
k5 = 170500 # seat post
k6 = 150000 #handle bars           ############
k7 = 390000 # human body stiffness 
k8 = 300000 # arms stiffness 
#Dampers
c1= 1100 # rear damper  using the equation c = (2*zeta*m*omega_natural)
c2= 150  #seat damping
c3 = 4000 #S
c4 = 4000
#Masses
m1=2 #Rear Wheel
m2=4 #Bike Frame                 ##############
m3=1.5 #Front Wheel
m4=1 #Seat Post (tube 348g)(saddle 520g)
m5=1.5 #handlebars 
m6=  32.15  #25.05  #42.58  #humanbody, larger portion ##############
m7=  14.668  #12.28 #17.28  #arms portion of body    ######
I_i=5 #
#Bike dimensions
L1= 0.5#distance between the centre of gravity and the rear wheel axis
L2= 0.5 #distance between the centre of gravity and the front wheel axis
s= 0.219 #distance between the centre of gravity and the seatpost axis
H= 0.481 #distance between the centre of gravity and the handlebar axis

#Degree-of-freedom
dof = 8
time_step = 0.0001
end_time = 20
# n = int(end_time/time_step)
time = [round(t,5) for t in np.arange(0, end_time, time_step) ]
velocity = 5 #m/s
###########################################################
###########################################################
###########################################################
#creating the road profile of for different grades of roads (file:///C:/Users/cbth/Downloads/ASCEStructuresCongress2008RoadRoughness.pdf)
    
    
g_d = [16, 64, 256] #PSD value at a reference number of omega_0, depends on road class
omega_0 = 0.1 #value of 0.1 rad/m
m = 2



time = np.arange(0,end_time,time_step)
time2 = np.arange(0,end_time-(time_step),time_step)

lamda = np.arange(20.0001, 0.0001, -1e-4) #spectrum of wavelengths to consider
N = len(lamda)


T1 = []
for i in range(1, N+1):
    theta_new = np.random.normal(2*np.pi, np.pi)
    theta = round(theta_new, 4)/2
    T1.append(theta)
    
    
O1 = []
O2 = []
for t in np.arange(0, N, 1):
    omega_new = ((velocity*2*np.pi)/lamda[t])#rad/s 
    omega_new0 = (2*np.pi)/lamda[t]
    omega = omega_new
    omega_0 = omega_new0
    O1.append(omega)
    O2.append(omega_0)
  

# O_0 = 0.16
# P1 = []
# for t in np.arange(0, N, 1):
#     if O1[t] < O_0:   
#         phi_new =  g_d[2]*0.000001 * ((O1[t])/O_0)**(-2)
#         phi = round(phi_new,10)
#         P1.append(phi)     
#     else:
#         if O_0 <= O1[t]:
#             phi_new =  g_d[2]*0.000001 * ((O1[t])/O_0)**(-1.5)
#             phi = round(phi_new,10)
#             P1.append(phi)
            
P1 = []
for t in np.arange(0, N, 1):
    phi_new = g_d[2]*0.000001 * ((O2[t])/1)**(-m) # ([rad/m] / [rad/m])
    phi = round(phi_new, 4)
    P1.append(phi)


D1 =[]
for i in np.arange(0, N, 1):
    d_omega_new = ((O1[N-1] - O1[i]) / (N - 1))  #rad/s
    d_omega = d_omega_new
    D1.append(d_omega)


A1 =[]
for i in np.arange(0, N, 1):
    A_new = ((P1[i] * (D1[i]/np.pi)))**0.5
    A = A_new
    A1.append(A)

random.shuffle(A1)

Z1 = []
for i in np.arange(0, N, 1):
    Z_new = A1[i] * np.sin(((O1[i]*time[i]*i) - T1[i]))
    Z = Z_new  #multiplying by 2pi/60 converts the units to rad
    Z1.append(Z)

T = 20         # Sample Period
fs = 10000      # sample rate, Hz
cutoff = 2000 # desired cutoff frequency of the filter,Hz ,  slightly higher than actual 1.2 Hz
nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples

y = butter_lowpass_filter(Z1, cutoff, fs, order)

filteredforce = []
i=0
while i < 200000:
    I_n = k4*y[i]
    filteredforce.append(I_n)
    i = i + 1
  
# setup matrices
#Mass matrix
M = np.array([
    [m1, 0, 0, 0, 0, 0, 0, 0],
    [0, m2, 0, 0, 0, 0, 0, 0],
    [0, 0, m3, 0, 0, 0, 0, 0],
    [0, 0, 0, m4, 0, 0, 0, 0],
    [0, 0, 0, 0, m5, 0, 0, 0],
    [0, 0, 0, 0, 0, m6, 0, 0],
    [0, 0, 0, 0, 0, 0, m7, 0],
     [0, 0, 0, 0, 0, 0, 0, I_i]
    ])

##Damping matrix
C = np.array([
  [c1,    -c1, 0,     0, 0,   0, 0,  (c1*L1)], 
  [-c1, c1+c2, 0,   -c2, 0,   0, 0,  ((-c2*s)) - (c1*L1)],
  [0,           0, 0,     0, 0,   0, 0,  0],
  [(0),     -c2, 0, c2+c3, 0, -c3, 0, ((+c2*s))],
  [(0),       (0), 0,     0, c4,  0,-c4, 0],
  [(0),       (0), 0,   -c3, 0,  c3, 0,  0],
  [(0),       (0), 0,   0, -c4,  0, c4,  0],
  [ c1*L1,     -(c2*s) - c1*L1  , 0,   (c2*s), 0,  0, 0,   +((c2*s**2)) + (c1*L1**2)]
  ])

#Stiffness Matrix
K = np.array([
    [k1+k2,           -k2,     0,     0,     0,   0,   0,   (k2*L1)], 
    [  -k2,   k2+k4+k5+k6,   -k4,   -k5,   -k6,   0,   0,-(k2*L1)-((k5*s))+(k4*L2)+((k6*H)) ],
    [    0,           -k4, k3+k4,     0,     0,   0,   0, -(k4*L2)],
    [    0,           -k5,     0, k5+k7,     0, -k7,   0,+((k5*s)) ],
    [    0,           -k6,     0,     0, k6+k8,   0, -k8, (-(k6*H))],
    [    0,             0,     0,   -k7,     0,  k7,   0,0],
    [    0,             0,     0,     0,   -k8,   0,  k8,0],
    [   +(k2*L1), -(k5*s) - (k2*L1) + (k4*L2) +(k6*H),    -(k4*L2),     +(k5*s),   -(k6*H),   0,  0,  +(k5*s**2) + (k2*L1**2) + (k4*L2**2) +((k6*H**2))]
    ])

#Identity Matrix
I = np.identity(dof)
#State space variables
A = np.zeros((2*dof,2*dof))
B = np.zeros((2*dof,2*dof))
Y = np.zeros((2*dof,1))
#State-space form, matrices A and B
A[0:8,0:8] = M
A[8:16,8:16] = I

B[0:8,0:8] = C
B[0:8,8:16] = K
B[8:16,0:8] = -I

#find natural frequencies and mode shapes,
#eigh command works only for symmetric matrices. Requires change
evals, evecs = eigh(K, M)
frequencies = np.sqrt(evals)
print(frequencies)
print(evecs)

A_inv = inv(A)
X1 = []
X2 = []
X3 = []
X4 = []
X5 = []
X6 = []
X7 = []
X8 = []
####################
V1 = []
V2 = []
V3 = []
V4 = []
V5 = []
V6 = []
V7 = []
V8 = []

#F = np.zeros((2*dof,1))
force1 = []
#pyt = []
t = 0
i = 0
while i < 200000:
    Y_new = Y + RK4_step(Y, t, time_step,i)
    Y = Y_new
    force1.extend(F(t,i)[2])
    t = round(t + time_step, 6)
    i = i + 1
    print(t)
    print(i)
    print(Y)
    
 #
    X1.extend(Y[8])
    X2.extend(Y[9])
    X3.extend(Y[10])
    X4.extend(Y[11])
    X5.extend(Y[12])
    X6.extend(Y[13])
    X7.extend(Y[14])
    X8.extend(Y[15])
    #    
    V1.extend(Y[0])
    V2.extend(Y[1])
    V3.extend(Y[2])
    V4.extend(Y[3])
    V5.extend(Y[4])
    V6.extend(Y[5])
    V7.extend(Y[6])
    V8.extend(Y[7])

G1 = []
G2 = []
G3 = []
G4 = []
G5 = []
G6 = []
G7 = []
G8 = []

#Finding the accelerations corresponding to each body
G1 = np.gradient(V1, time_step)
G2 = np.gradient(V2, time_step)
G3 = np.gradient(V3, time_step)
G4 = np.gradient(V4, time_step)
G5 = np.gradient(V5, time_step)
G6 = np.gradient(V6, time_step)
G7 = np.gradient(V7, time_step)
G8 = np.gradient(V8, time_step)





plt.plot(time, filteredforce, color ='g', LineWidth = 0.1, label = 'Original')
plt.xlabel('time (s)')
plt.ylabel('force (m)')
plt.title('FILTERED force')
plt.show()

############RMS#####################

A1_rms = (sum(G6**2)/N)**0.5
A2_rms = (sum(G7**2)/N)**0.5 #probably these values will need to be filtered so that a proper average can be taken

X1_rms = (sum(np.array(X6)**2)/N)**0.5
X2_rms = (sum(np.array(X7)**2)/N)**0.5


Z_rms = (sum(np.array(Z1)**2)/N)**0.5

##########################################################
##########################################################


F2 = np.array(O1)/(2*np.pi)  #converting from rad/s to Hz

#calculating the PSD AND FFT
fhat = np.fft.fft(G6,N)
fhat2 = np.fft.fft(G7,N)
ifhat = fhat * np.conj(fhat)
ifhat2 = fhat2*np.conj(fhat2)
problem = (ifhat)/N
problem2 = (ifhat2)/N
freq = (1/(time_step*N))*np.arange(N)
L = np.arange(1,np.floor(N), dtype='int')


G6_max = max(ifhat)
G6_min = min(ifhat)

G6_PSD_norm = []
for i in np.arange(0,N,1):
    G_norm_new = (ifhat[i] - G6_min)/(G6_max-G6_min)
    G6_norm = G_norm_new
    G6_PSD_norm.append(G6_norm)


plt.style.use('classic')
Figure3 = plt.figure()
plt3 = Figure3.add_subplot(221)
plt4 = Figure3.add_subplot(223)
plt3.plot(O1, problem, color = 'g')
plt4.plot(O1, problem2, color= 'b')
plt3.set_title('PSD-Acceleration')
Figure3.subplots_adjust(hspace=0.2)
plt.show()


plt.plot(F2, problem, color = 'g')
plt.plot(F2, problem2, color= 'b')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.xlim(0,80)
plt.title('PSD ')
plt.show()


plt.plot(time, Z1, color ='g', LineWidth = 0.1, label = 'Original')
plt.xlabel('time (s)')
plt.ylabel('displacement (m)')
plt.title('Road Profile')
plt.xlim(0,20)
plt.ylim(-0.05, 0.05)
plt.show()

#########################################################
#########################################################
plt.style.use('classic')
Figure1 = plt.figure()
plt1 = Figure1.add_subplot(221)
plt2 = Figure1.add_subplot(223)
plt1.plot(time, X4, color = 'g')
plt2.plot(time, X5, color= 'b')
plt1.set_title('Displacements')
Figure1.subplots_adjust(hspace=0.2)
plt.show()
###############
#Accelerations#
######plotting the accelerations of different bodies of the system

plt.plot(time, G6)
plt.plot(time, G7)
plt.xlim(0, 1)
plt.title('Acceleration Response (m/s^2)')
plt.xlabel('time (s)')
plt.ylabel('Acceleration (ms-2)')
plt.legend(['A1','A2','A3','A4','A5','A6','A7'], loc='lower right')
plt.show()
##########


#Plotting the displacements of the different bodies of the system
plt.plot(time,X6)
plt.plot(time,X7)
plt.xlabel('time (s)')
plt.ylabel('displacement (m)')
plt.title('Response Curves')
plt.legend(['X6','X7'], loc='lower right')
plt.show()

#####

Force_plot = plt.plot(time,force1, color = 'g', LineWidth = 0.1, label = 'force1')
plt.ylim(-1000,1000)
plt.xlabel('time (s)')
plt.ylabel('Force1 (N)')
plt.title('Force Exerted by Road Profile, N')
plt.show()


         
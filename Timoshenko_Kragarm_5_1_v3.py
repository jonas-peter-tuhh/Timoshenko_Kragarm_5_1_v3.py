"""
Created on Mon Aug 29 17:20:05 2022
@author: Jonas Peter
"""
##
import torch
import torch.nn as nn
from torch.autograd import Variable
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
import torch.autograd
from scipy.optimize import curve_fit

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train = True
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer1 = nn.Linear(1, 5)
        #nn.BatchNorm1d(5)
        self.hidden_layer2 = nn.Linear(5, 10)
        #nn.BatchNorm1d(15)
        self.hidden_layer3 = nn.Linear(10, 10)
        #nn.BatchNorm1d(50)
        self.hidden_layer4 = nn.Linear(10, 10)
        #nn.BatchNorm1d(50)
        self.hidden_layer5 = nn.Linear(10, 10)
        #nn.BatchNorm1d(50)
        self.hidden_layer6 = nn.Linear(10, 10)
        #nn.BatchNorm1d(25)
        self.hidden_layer7 = nn.Linear(10, 10)
        #nn.BatchNorm1d(15)
        self.output_layer = nn.Linear(10, 2)

    def forward(self, x):
        inputs = x
        layer1_out = torch.tanh(self.hidden_layer1(inputs))
        layer2_out = torch.tanh(self.hidden_layer2(layer1_out))
        layer3_out = torch.tanh(self.hidden_layer3(layer2_out))
        layer4_out = torch.tanh(self.hidden_layer4(layer3_out))
        layer5_out = torch.tanh(self.hidden_layer5(layer4_out))
        layer6_out = torch.tanh(self.hidden_layer6(layer5_out))
        layer7_out = torch.tanh(self.hidden_layer7(layer6_out))
        output = self.output_layer(layer7_out)
        return torch.unsqueeze(output.reshape(-1),1)
##
choice_load = input("Möchtest du ein State_Dict laden? y/n")
if choice_load == 'y':
    train=False
    filename = input("Welches State_Dict möchtest du laden?")
    net = Net()
    net = net.to(device)
    net.load_state_dict(torch.load('C:\\Users\\Administrator\\Desktop\\Uni\\Master\\Masterarbeit\\Timoshenko_Kragarm_5.1_v2\\saved_data\\'+filename))
    net.eval()
##
# Hyperparameter
learning_rate = 0.1
if train:
    net = Net()
    net = net.to(device)
mse_cost_function = torch.nn.MSELoss()  # Mean squared error
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# Definition der Parameter des statischen Ersatzsystems

Lb = float(input('Länge des Kragarms [m]: '))
E = 21#float(input('E-Modul des Balkens [10^6 kNcm²]: '))
h = 10#float(input('Querschnittshöhe des Balkens [cm]: '))
b = 10#float(input('Querschnittsbreite des Balkens [cm]: '))
A = h*b
I = (b*h**3)/12
EI = E*I*10**-3
G = 80#float(input('Schubmodul des Balkens [GPa]: '))
LFS = 1#int(input('Anzahl Streckenlasten: '))
K = 5 / 6  # float(input(' Schubkoeffizient '))
Ln = np.zeros(LFS)
Lq = np.zeros(LFS)
s = [None] * LFS
normfactor = 10/(Lb**3/(K*A*G)+(11*Lb**5)/(120*EI))

#Der Scheduler sorgt dafür, dass die Learning Rate auf einem Plateau mit dem factor multipliziert wird
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, verbose=True, factor=0.7)

for i in range(LFS):
    # ODE als Loss-Funktion, Streckenlast
    Ln[i] = 0#float(input('Länge Einspannung bis Anfang der ' + str(i + 1) + '. Streckenlast [m]: '))
    Lq[i] = Lb#float(input('Länge der ' + str(i + 1) + '. Streckenlast [m]: '))
    s[i] = str(normfactor)+"*x"#input(str(i + 1) + '. Streckenlast eingeben: ')

def h(x, j):
    return eval(s[j])

#Netzwerk System 1
def f(x, net):
    net_out = net(x)
    phi = net_out[1::2]
    phi_x = torch.autograd.grad(phi, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(phi))[0]
    v = net_out[0::2]
    v_x = torch.autograd.grad(v, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(v))[0]
    v_xx = torch.autograd.grad(v_x, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(v))[0]
    ode = 0
    for i in range(LFS):
        ode += ((K*A*G) * (v_xx - phi_x) - h(x - Ln[i], i)) * (x <= (Ln[i] + Lq[i])) * (x >= Ln[i])
    return ode

def g(x, net):
    net_out = net(x)
    ode = 0
    phi = net_out[1::2]
    phi_x = torch.autograd.grad(phi, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(phi))[0]
    phi_xx = torch.autograd.grad(phi_x, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(phi))[0]
    v = net_out[0::2]
    v_x = torch.autograd.grad(v, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(v))[0]
    for i in range(LFS):
        ode += (-EI*phi_xx + (K*A*G)*(v_x - phi)) * (x <= (Ln[i] + Lq[i])) * (x >= Ln[i])
    return ode


x = np.linspace(0, Lb, 1000)
pt_x = torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=True).to(device), 1)
qx = np.zeros(1000)
for i in range(LFS):
    qx = qx + (h(torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=False).to(device), 1) - Ln[i], i).cpu().detach().numpy()).squeeze() * (x <= (Ln[i] + Lq[i])) * (x >= Ln[i])

Q0 = integrate.cumtrapz(qx, x, initial=0)
#Q0 = Q(0) = int(q(x)), über den ganzen Balken
qxx = qx * x
#M0 = M(0) = int(q(x)*x), über den ganzen Balken
M0 = integrate.cumtrapz(qxx, x, initial=0)
#Die nächsten Zeilen bis Iterationen geben nur die Biegelinie aus welche alle 10 Iterationen refreshed wird während des Lernens, man kann also den Lernprozess beobachten
if train:
    y1 = net(torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=False).to(device), 1))
    fig = plt.figure()
    plt.grid()
    ax1 = fig.add_subplot()
    ax1.set_xlim([0, Lb])
    ax1.set_ylim([-20, 0])
    #ax2.set_
    net_out_plot = y1.cpu().detach().numpy()
    line1, = ax1.plot(x, net_out_plot[0::2])
    f_anal = (-1/120 * normfactor *pt_x**5 + 1/6 * Q0[-1] * pt_x**3 - M0[-1]/2 *pt_x**2)/EI + (1/6 * normfactor * (pt_x)**3 - Q0[-1]*pt_x)/(K*A*G)
##
iterations = 1000000
for epoch in range(iterations):
    if not train: break
    optimizer.zero_grad()  # to make the gradients zero
    x_bc = np.linspace(0, Lb, 5000)
    # linspace x Vektor zwischen 0 und 1, 500 Einträge gleichmäßiger Abstand
    # Zufällige Werte zwischen 0 und 1
    pt_x_bc = torch.unsqueeze(Variable(torch.from_numpy(x_bc).float(), requires_grad=True).to(device), 1)
    # unsqueeze wegen Kompatibilität
    pt_zero = Variable(torch.from_numpy(np.zeros(1)).float(), requires_grad=False).to(device)

    x_collocation = np.random.uniform(low=0.0, high=Lb, size=(250 * int(Lb), 1))
    all_zeros = np.zeros((250 * int(Lb), 1))

    pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
    ode_1 = f(pt_x_collocation, net)
    ode_2 = g(pt_x_collocation, net)

    # Randbedingungen
    net_bc_out = net(pt_x_bc)

    # Netzwerkausgabewerte berechnen
    phi = net_bc_out[1::2]
    v = net_bc_out[0::2]

    #für phi:
    phi_x = torch.autograd.grad(phi, pt_x_bc, create_graph=True, retain_graph=True,
                                grad_outputs=torch.ones_like(phi))[0]
    v_x = torch.autograd.grad(v, pt_x_bc, create_graph=True, retain_graph=True,
                                grad_outputs=torch.ones_like(v))[0]

    bc1 = phi[0]
    bc2 = phi_x[0] + M0[-1]/EI
    bc3 = phi_x[-1]
    bc4 = (v_x[0] - phi[0]) + Q0[-1]/(K*A*G)
    bc5 = (v_x[-1] - phi[-1])
    bc6 = v[0]


    #Alle e's werden gegen 0-Vektor (pt_zero) optimiert.
    #BC Error
    mse_bc = mse_cost_function(bc1, pt_zero) + 10*mse_cost_function(bc2, pt_zero) + mse_cost_function(bc3, pt_zero) + 1000000*mse_cost_function(bc4, pt_zero) + mse_cost_function(bc5, pt_zero) + mse_cost_function(bc6, pt_zero)
    #ODE Error
    mse_ode_1 = 1/normfactor *mse_cost_function(ode_1, pt_all_zeros)
    mse_ode_2 = 1/normfactor *mse_cost_function(ode_2, pt_all_zeros)

    loss = mse_ode_1 + mse_ode_2 + mse_bc
    loss = 1/normfactor * loss

    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    with torch.autograd.no_grad():
        if epoch % 10 == 9:
            print(epoch, "Traning Loss:", loss.data)
            plt.grid()
            net_out  = net(pt_x)
            net_out_v = net_out[1::2]
            net_out_v_cpu = net_out_v.cpu().detach().numpy()
            err = torch.norm(net_out_v-f_anal,2)
            print(f'Error = {err}')
            if err < 0.1*Lb:
                print(f"Die L^2 Norm des Fehlers ist {err}.\nStoppe Lernprozess")
                break
            line1.set_ydata(net_out_v_cpu)
            fig.canvas.draw()
            fig.canvas.flush_events()

##
if choice_load == 'n':
    choice_save = input("Möchtest du die Netzwerkparameter abspeichern? y/n")
    if choice_save == 'y':
        filename = input("Wie soll das State_Dict heißen?")
        torch.save(net.state_dict(),'C:\\Users\\Administrator\\Desktop\\Uni\\Master\\Masterarbeit\\Timoshenko_Kragarm_5.1_v2\\saved_data\\'+filename)
## Plots für Kapitel 5.1
x = np.linspace(0, Lb, 1000)


pt_u_out = net(pt_x)
v_out = pt_u_out[0::2]

v_out_x = torch.autograd.grad(v_out, pt_x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(v_out))[0]
v_out_x = v_out_x.cpu().detach().numpy()

#Vges Kompatibilität Numpy Array
net_out_cpu = pt_u_out.cpu()
net_out = net_out_cpu.detach()
net_out_np = net_out.numpy()

v_out = net_out_np[0::2]
phi_out = net_out_np[1::2]
fig = plt.figure()


plt.subplot(2, 2, 1)
plt.title('$v_{ges}$ Auslenkung')
plt.xlabel('')
plt.ylabel('[cm]')
plt.plot(x, v_out)
plt.plot(x, ((-1/120 * normfactor * x**5 + Q0[-1]/6 * x**3 - M0[-1]/2 * x**2)/EI + (1/6 * normfactor* x**3 - Q0[-1] * x)/(K*A*G)))
plt.grid()

plt.subplot(2, 2, 3)
plt.title('$\phi$ Neigung')
plt.xlabel('')
plt.ylabel('$10^{-2}$')
plt.plot(x, (phi_out))
plt.plot(x, (-1/24 * normfactor * x**4 + 0.5 * Q0[-1] * x**2 - M0[-1] * x)/EI)
plt.grid()


plt.subplot(2, 2, 2)
plt.title('Schubwinkel $\gamma$')
plt.xlabel('')
plt.ylabel('$(10^{-2})$')
plt.plot(x, ((v_out_x-phi_out)*EI)/(K*A*G))
plt.plot(x, ((normfactor * 0.5 * x**2 - Q0[-1])/(K*A*G)))
plt.grid()

gamma_anal = ((normfactor * 0.5 * x**2 - Q0[-1])/(K*A*G))
gamma_net = v_out_x-phi_out
gamma_err = np.linalg.norm((gamma_net-gamma_anal), 2)/np.linalg.norm(gamma_anal, 2)
print('\u03B3 5.1 =',gamma_err)

plt.show()


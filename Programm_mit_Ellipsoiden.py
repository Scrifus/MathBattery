# ---------------- Simulationseinstellungen ----------------

info = 4 # Info-Level der Ausgabe, kann einen Wert zwischen 0-5 annehmen
N = 101 # Anzahl Intervallpunkte
T_sim = 20000 # Dauer der Simulation in Sekunden
tau_dimensioniert = 0.5 # Zeitschrittweite der Simulation in Sekunden
N_solid = 30 # Anzahl der Intervallpunkte in den Aktivpartikeln
SOC = 0.1 # Der Ladezustand der Zelle in Prozent zum Start der Simulation
J_ext = 10 # Externe Stromdichte an der Zelle in A/m², positive für Ladung, negativ für Entladung
plot_frequency  = 20 # Anzahl der Zwischenwerte für die Plots


# ---------------- Imports ----------------

import time
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve
from scipy.sparse import diags, lil_matrix
from databaseLGM50 import case_1 as database
from databaseLGM50 import Ubsp, Ubsn
from scipy.sparse import lil_matrix, csc_matrix
import montecarlo_anode as mc_a
import montecarlo_kathode as mc_k
import numpy as np
from scipy.optimize import fsolve
from scipy.sparse.linalg import splu
import matplotlib as mpl
import matplotlib.pyplot as plt 
from scipy.stats import rv_discrete, norm
"""
mpl.rcParams.update({
    "font.size": 20,        # Basisschrift
    "axes.titlesize": 20,   # Achsentitel
    "axes.labelsize": 20,   # x/y-Label
    "xtick.labelsize": 12,  # x-Ticks
    "ytick.labelsize": 12,  # y-Ticks
    "legend.fontsize": 20   # Legende
})
"""
# ---------------- Physikalische Konstanten ----------------
spezifische_Oberflaeche_anode = mc_a.spezifische_Oberflaeche
spezifische_Oberflaeche_kathode = mc_k.spezifische_Oberflaeche

Umfang_anode = mc_a.Umfang
Umfang_kathode = mc_k.Umfang

K_anode = mc_a.K
K_kathode = mc_k.K

M_anode = mc_a.M
M_kathode = mc_k.M

F_anode = mc_a.F
F_kathode = mc_k.F
D_e = database["difen"]
kappa_e = database["kappa_e"]
kappa_s_n = database["sigma_n"]
kappa_s_p = database["sigma_p"]
d_s_n = database["difsn"]
d_s_p = database["difsp"]
Faraday = database["Fara"]
T_ref = database["T_ref"]
i00 = database["i00"]
Xi = database["Xi"]

L_p_l = database["L_p"]
L_n_l = database["L_n"]
R_p = database["R_p"]
R_n = database["R_n"]
asn = database["asn"]
asp = database["asp"]
t_plus = database["t_plus"]

C_ref = database["C_ref"]
cmaxn = database["cmaxn"]
cmaxp = database["cmaxp"]
U_ref = database["U_ref"]
 
sigma_n = database["sigma_n"]
sigma_p = database["sigma_p"]

StoichAt0_p   = database["StoichAt0_p"]
StoichAt100_p = database["StoichAt100_p"]
StoichAt0_n   = database["StoichAt0_n"]
StoichAt100_n = database["StoichAt100_n"]


T = T_sim/T_ref
J_ext = J_ext / database["J_ref"]
tau = tau_dimensioniert/T_ref # Zeitschrittweite entdimensioniert
steps = int(T / tau)
h = 1 / (N - 1)
L_n = round(L_n_l/h+1)
L_p = round((1-L_p_l)/h)
x = np.linspace(0, 1, N) # Spatial mesh
y = np.linspace(0, 1, N_solid)# Radial mesh
print_frequency = round(steps/plot_frequency)

cs0n =  (StoichAt0_n+(SOC*(StoichAt100_n-StoichAt0_n))) * cmaxn        # CHEN2020: Stoichiometry at 100%/0% Table VII Negative electrode
cs0p = (StoichAt0_p-(SOC*(StoichAt0_p-StoichAt100_p))) * cmaxp         # CHEN2020: Stoichiometry at 100%/0% Table VII Positive electrode

if info > 1:
    print("Program potential_solid_3D_11.01")
    print(f"  N  = {N:4d} ",f"  N_n = {L_n:3d} ",f"  N_p = {N-L_p:3d} ",f"  N_sep = {L_p-L_n:3d} ",f"  N_solid = {N_solid:2d} ")
    print(f"  Ln = {L_n_l:4.2f} ",f"  Lp  = {L_p_l:4.2f} ")
    print(f"  T  = {(T*T_ref):4.2f}s ",f"  tau = {(tau*T_ref):7.2e}s ")

# ---------------- Anfangswerte ----------------
# Konzentration
ce = np.full(N,database["ce0"])
cs = np.zeros(N)
 # 1) Anoden-Region [0 .. L_n]:
cs[:L_n] = cs0n

 # 2) Kathoden-Region [L_p .. N-1]:
cs[L_p:] = cs0p
matrix_cs = [np.full(N_solid, cs[i]) for i in range(N)]

# Potentiale
pe = np.log(ce) / 2

ps_start = np.zeros(N)
ps_start[:L_n] = 0.0
U_cell_guess = Ubsp(cs[L_p:].mean() / cmaxp)/U_ref * 39.6  # 3.2V in dimensionslos
ps_start[L_p:] = U_cell_guess
ps = ps_start.copy()

test = np.zeros(N)
length = len(test[L_p:])

# Lokale Groessen
idx = np.arange(len(cs))
mask_n = idx < L_n        # Anodenregion
mask_p = idx >= L_p        # Kathodenregion
mask_sep = (idx >= L_n) & (idx < L_p)

# Lokale c_max je nachdem, ob Anode oder Kathode
cmax = np.empty_like(cs)
cmax[mask_n] = cmaxn
cmax[mask_p] = cmaxp
cmax[mask_sep] = 0 # Im Separator keine Reaktion, dh keine Begrenzung noetig


def discrete_normal_dist(mu=10, sigma=3, low=0, high=20):
    xs = np.arange(low, high + 1)
    pdf_vals = norm.pdf(xs, loc=mu, scale=sigma)
    pmf = pdf_vals / pdf_vals.sum()
    return rv_discrete(values=(xs, pmf))

dist = discrete_normal_dist()

#HIER wird der N-lange Vektor erzeugt

Verteilung = np.asarray(dist.rvs(size=N), dtype=int)
# Flaechenfaktor a_lokal
a_lokal = np.zeros_like(cs)
Verteilung_n = Verteilung[mask_n]
Verteilung_p = Verteilung[mask_p]

# Flaechenfaktor a_lokal (Skalar pro Punkt)
a_lokal = np.zeros(N, dtype=float)
a_lokal[mask_n] = spezifische_Oberflaeche_anode[Verteilung_n]
a_lokal[mask_p] = spezifische_Oberflaeche_kathode[Verteilung_p]
asn=a_lokal[mask_n]
asp=a_lokal[mask_p]

# Umfang (Skalar pro Punkt)
Umfang_lokal = np.zeros(N, dtype=float)
Umfang_lokal[mask_n] = Umfang_anode[Verteilung_n]
Umfang_lokal[mask_p] = Umfang_kathode[Verteilung_p]

p = 10 # Anzahl der Basisfunktionen

# M und K: (N,10,10)
M_lokal = np.zeros((N, p, p), dtype=float)
K_lokal = np.zeros((N, p, p), dtype=float)
M_lokal[mask_n, :, :] = M_anode[Verteilung_n]
M_lokal[mask_p, :, :] = M_kathode[Verteilung_p]
K_lokal[mask_n, :, :] = K_anode[Verteilung_n]
K_lokal[mask_p, :, :] = K_kathode[Verteilung_p]

# F: (N,10)
F_galerkin_lokal = np.zeros((N, p), dtype=float)
F_galerkin_lokal[mask_n, :] = F_anode[Verteilung_n]
F_galerkin_lokal[mask_p, :] = F_kathode[Verteilung_p]

# Start-Koeffizienten a (N,10)
Gesamtkoeffizienten = np.zeros((N, p), dtype=float)
Gesamtkoeffizienten[mask_n, 0] = cs0n
Gesamtkoeffizienten[mask_p, 0] = cs0p

# ---------------- Funktionen ----------------

# Butler-Volmer Funktion 
def i_BV(ce, ue, cs, us, Elektrode = None):
    """
    Butler-Volmer-Stromdichte mit Stöchiometrie-basiertem OCP.
    """

    if Elektrode == 'n':
        x_n = cs / cmaxn
        # Austauschstrom i0
        i0 = np.sqrt(ce * cs * (cmaxn - cs))
        i0 = np.where(cs > cmaxn, 0.0, i0)
        i_BV = 2.0 * i00 * i0 * np.sinh((us - ue - Ubsn(x_n)) / 2)
    elif Elektrode == 'p':
        x_p = cs / cmaxp
        # Austauschstrom i0
        i0 = np.sqrt(ce * cs * (cmaxp - cs))
        i0 = np.where(cs > cmaxp, 0.0, i0)
        i_BV = 2.0 * i00 * i0 * np.sinh((us - ue - Ubsp(x_p)) / 2)
    else:
        # Austauschstrom i0
        i0 = np.sqrt(ce * cs * (cmax - cs))
        i0 = np.where(cs > cmax, 0.0, i0)
        # Ueberpotential je nach Region
        ubs = np.zeros_like(cs)
        ubs[mask_n] = Ubsn(cs[mask_n] / cmaxn)
        ubs[mask_p] = Ubsp(cs[mask_p] / cmaxp)
        # Im Separator bleibt ubs=0
 
        # Butler-Volmer-Gleichung
        i_BV = 2.0 * i00 * i0 * np.sinh((us - ue - ubs) / 2)

    return i_BV

# Matrizenberechnung
def A_u():
    A = np.zeros((N, N))
    main_diag_u = 2 * kappa_e / h**2 * np.ones(N)
    off_diag_u  = -(kappa_e / h**2) * np.ones(N - 1)
    A = diags([off_diag_u, main_diag_u, off_diag_u], [-1, 0, 1]).tocsc()
    # Randbedingungen:
    A[ 0,  1] = -2 * kappa_e / h**2 # Neumann links
    A[-1, -2] = -2 * kappa_e / h**2 # Neumann rechts
    return A
 
def A_ps_n():
    A = lil_matrix((L_n, L_n))
    for i in range(1, L_n-1):
        A[i, i-1] = -sigma_n / h**2
        A[i, i]   = 2 * sigma_n / h**2
        A[i, i+1] = -sigma_n / h**2
    # Randbedingungen:
    A[0, 1] = 0 # Dirichlet links
    A[0, 0] = 1
    A[-1, -2] = -2 * sigma_n / h**2 # Neumann rechts
    A[-1, -1] =  2 * sigma_n / h**2
    return csc_matrix(A)
 
def A_ps_p():
    A = lil_matrix((length, length))
    for i in range(1, length-1):
        A[i, i-1] = -sigma_p / h**2
        A[i, i]   = 2 * sigma_p / h**2
        A[i, i+1] = -sigma_p / h**2
    # Randbedingungen:
    A[0, 1] = -2 * sigma_p / h**2 # Neumann links
    A[0, 0] =  2 * sigma_p / h**2
    A[-1, -2] = 0 # Dirichlet rechts
    A[-1, -1] = 1
    return csc_matrix(A)

def A_c():
    A = np.zeros((N, N))
    haupt_diag_c = (1 + 2 * D_e * tau / h**2) * np.ones(N)
    neben_diag_c = -(D_e * tau / h**2) * np.ones(N - 1)
    A = diags([neben_diag_c, haupt_diag_c, neben_diag_c], [-1, 0, 1]).tocsc()
    # Randbedingungen:
    A[ 0,  1] = -2 * D_e * tau / h**2 # Neumann links
    A[-1, -2] = -2 * D_e * tau / h**2 # Neumann rechts
    return A


def berechne_ps(ce, pe, cs, ps):
    ps_n = ps[:L_n]
    ce_n = ce[:L_n]
    pe_n = pe[:L_n]
    cs_n = cs[:L_n]
    rhs_g =  F_Potenzial_Solid(ce_n, pe_n, cs_n, ps_n, Elektrode = 'n') + G_Potenzial_Solid_t_0[:L_n]
    rhs_g[0] = 0
    ps_n = A_ps_n_splu.solve(rhs_g)
    ps_p_0 = ps[L_p:]
    pe_p = pe[L_p:]
    ce_p = ce[L_p:]
    cs_p = cs[L_p:]
   
    def f_ps_p(ps_p):
        return F_Potenzial_Solid(ce_p, pe_p, cs_p, ps_p, Elektrode= 'p')
   
    ps_p = bisektionsverfahren(A_ps_p_matrix, f_ps_p, ps_p_0, h, J_ext)
    ps_new = np.concatenate([
        ps_n,
        np.zeros(L_p - L_n), # Separator
        ps_p
    ])
    return ps_new

def bisektionsverfahren(A, f, u0, h, I0):
    Startwert = u0[-1]
    bisektionswerte = np.full(2, Startwert)

    def Matrixloeser(u_rand, f):
        u = np.full(length, u_rand)
        for i in range(100):
            rhs = f(u)
            rhs[-1] = u_rand
            u_neu = A_ps_p_splu.solve(rhs)
            if mag(u_neu - u) < 1e-10:
               #if info > 4:
                  #print(f"   Matrixloeser: {i:3d}")
               break
            u = u_neu
        return u

    def Integral_wert(phi_rand):
        phi_g = Matrixloeser(phi_rand, f)
        return np.trapezoid(-f(phi_g), dx=h) - I0

    
    fa = fb = None
    for i in range(100):
        bisektionswerte[0] = Startwert - 2**(i-2)
        bisektionswerte[1] = Startwert + 2**(i-2)
        a, b = bisektionswerte[0], bisektionswerte[1]
        fa = Integral_wert(a)
        fb = Integral_wert(b)
        if fa * fb < 0:
            break

    # Regula Falsi
    for i in range(100):
        a, b = bisektionswerte[0], bisektionswerte[1]

        
        if fa is None or a != bisektionswerte[0]:
            fa = Integral_wert(a)
        if fb is None or b != bisektionswerte[1]:
            fb = Integral_wert(b)

        # Schutz gegen Division durch 0
        if abs(fb - fa) < 1e-30:
            print("   Abbruch: fb - fa = 0 (Regula Falsi)")
            break

        # Schnittpunkt der Sekante mit der x-Achse
        xm = b - fb * (b - a) / (fb - fa)
        fm = Integral_wert(xm)

        if abs(fm) < 1e-10:
            if info > 5:
                print(f"   Regula Falsi Iterationen: {i:3d}")
            if fa * fm < 0:
                bisektionswerte[1] = xm
                fb = fm
            elif fb * fm < 0:
                bisektionswerte[0] = xm
                fa = fm
            break

       
        if fa * fm < 0:
            bisektionswerte[1] = xm
            fb = fm
        elif fb * fm < 0:
            bisektionswerte[0] = xm
            fa = fm
        else:
            print("   Fehler Regula Falsi (kein Vorzeichenwechsel)")
            break
    phi_p = Matrixloeser(xm, f)
    return phi_p



def f_u(u):
    phi_e = u + np.log(ce) / 2
    return G_Potenzial_t_0 + F_Potenzial(ce, phi_e, cs, ps)

def meansol(A: np.ndarray, f, u0: np.ndarray) -> np.ndarray:
    u0 = u0 - np.log(ce) / 2 # P_e zu U_e
    if hasattr(A, "toarray"):
        A = A.toarray()
    u0 = np.asarray(u0)
    N  = u0.size
    ev = np.ones(N)
    # 1) Mittelwert herausnehmen
    u0m = np.mean(u0)
    u0c = u0 - u0m
    # 2) Kompatibilitätsbedingung
    def compat(s):
        return np.trapezoid(f(u0c + s * ev),dx = h)
    # 3) Nullstelle von s->compat(s) finden
    u0m_root, = fsolve(compat, u0m, maxfev=100)
    #print(compat(u0m_root))
    # 4) Kompatible rechte Seite
    rhs = f(u0c + u0m_root * ev)
    # 5) Blocksystem 
    BM = np.block([
        [A,                ev.reshape(-1,1)],
        [ev.reshape(1,-1), np.zeros((1,1))]
    ])
    FM = np.concatenate([rhs, [0]])
    # 6) System loesen und u extrahieren
    UM = np.linalg.solve(BM, FM)
    u = UM[:N]
    # 7) Verschiebung zurueckaddieren
    u_return = u + u0m_root
    return u_return + np.log(ce) / 2 # U_e zu P_e

# Interface

def F_Konzentration(ce, pe, cs, ps):
    # Butler–Volmer-Strom
    ibv = i_BV(ce, pe, cs, ps)
    # Rückgabe der Reaktionsrate
    return (a_lokal / Faraday) * ibv * (1 - t_plus)

def F_Potenzial(ce, pe, cs, ps):
    """
    Potenzial-Quellenfunktion mit getrennten spezifischen Flaechen
    fuer Anode (asn) und Kathode (asp).
    """
    # Butler–Volmer-Strom
    ibv = i_BV(ce, pe, cs, ps)
    # Potenzialquelle
    return a_lokal * ibv
 
def F_Potenzial_Solid(ce, pe, cs, ps, Elektrode=None):
    """
    Solid-Potenzial-Quellenfunktion mit getrennten spezifischen Flaechen
    fuer Anode (asn) und Kathode (asp).
    """
    # Butler-Volmer-Strom im Solid
    ibv = i_BV(ce, pe, cs, ps, Elektrode)
    if Elektrode == 'n':
        a_gebiet = asn
    elif Elektrode == 'p':
        a_gebiet = asp
    else:
        a_gebiet = a_lokal
    return -a_gebiet * ibv
 
# Randbedingungen - teilweise momentan ungenutzt
def G_Konzentration():
    G = np.zeros(N)
    return G
 
def G_Potenzial(): # Nicht gebraucht
    G = np.zeros(N)
    G[0]  = 0
    G[-1] = 0
    return G
 
def G_Potenzial_t_0():
    G = np.zeros(N)
    G[0]  = 0
    G[-1] = 0
    return G
 
def G_Potenzial_Solid(): # Nicht gebraucht
    G = np.zeros(N)
    return G
 
def G_Potenzial_Solid_t_0():
    G = np.zeros(N)
    return G

def Flussrandbedingung(ce_s, ue_s, cs_s, ps_s):
    """
    Flussrandbedingung am aeussersten Punkt des Aktivpartikels.
    """
    return -(1.0 / Faraday) * i_BV(ce_s, ue_s, cs_s, ps_s)


def aktivpartikel(ce, pe, cs, ps):
    Randbedingung = Flussrandbedingung(ce, pe, cs, ps)
    cs_new = np.zeros_like(cs)

    # Anode
    for i in range(L_n):
        a_koeffizienten = Gesamtkoeffizienten[i]              # (10,)
        lhs_cs = (1/tau) * M_lokal[i] + K_lokal[i]            # (10,10)
        rhs_cs = (1/tau) * (M_lokal[i] @ a_koeffizienten) + Randbedingung[i] * F_galerkin_lokal[i]  # (10,)
        a_koeffizienten = solve(lhs_cs, rhs_cs)               # (10,)
        Summierter_Wert = np.dot(a_koeffizienten, F_galerkin_lokal[i])
        cs_new[i] = Summierter_Wert / Umfang_lokal[i]
        Gesamtkoeffizienten[i] = a_koeffizienten


    # Kathode
    for i in range(L_p, N):
        a_koeffizienten = Gesamtkoeffizienten[i]              # (10,)
        lhs_cs = (1/tau) * M_lokal[i] + K_lokal[i]            # (10,10)
        rhs_cs = (1/tau) * (M_lokal[i] @ a_koeffizienten) + Randbedingung[i] * F_galerkin_lokal[i]  # (10,)
        a_koeffizienten = solve(lhs_cs, rhs_cs)               # (10,)
        Summierter_Wert = np.dot(a_koeffizienten, F_galerkin_lokal[i])
        cs_new[i] = Summierter_Wert / Umfang_lokal[i]
        Gesamtkoeffizienten[i] = a_koeffizienten

    return cs_new

# Sonstiges

def Abbruchkriterium(pe, ps) :
    Integral_gesamt = np.trapezoid(F_Potenzial(ce, pe, cs, ps), dx = h)
    Integral_p = np.trapezoid(F_Potenzial(ce, pe, cs, ps)[L_p:], dx = h)
    if abs(Integral_gesamt) < 1e-6 and abs(Integral_p-J_ext) < 1e-6:
        return True
    else:
        return False
   
def mag(x):
    return math.sqrt(h * sum(z**2 for z in x)) 

# Vektoren
G_Potenzial_Solid_t_0 = G_Potenzial_Solid_t_0()
G_Potenzial_t_0 = G_Potenzial_t_0()
G_Konzentration = G_Konzentration()
 
# Matrizen
A_c_matrix_lu_zerlegung = splu(A_c().tocsc())
A_u_matrix = A_u()
A_ps_n_matrix = A_ps_n()
A_ps_p_matrix = A_ps_p()

A_ps_n_splu = splu(A_ps_n().tocsc())
A_ps_p_splu = splu(A_ps_p().tocsc())

# ---------------- Startpotenzial berechnen ----------------

 

def kombistartwert(ps_old, pe_old):
    global ps
    ps = ps_old
    for k in range(1000):
        for i in range(100):
            pe = meansol(A_u_matrix, f_u, pe_old)
            if mag(pe - pe_old) < 1e-10:
                if info > 4:
                    print("   Anfangspotential")
                    print("      Konvergenz in ", i, "Schritten zu |Pe-Pe_old| = ",mag(pe - pe_old))
                    break
            pe_old = pe
        for i in range(100):
            ps = berechne_ps(ce, pe, cs, ps_old)
            if mag(ps - ps_old) < 1e-10:
                if info > 4:
                    print("   Anfangspotential")
                    print("      Konvergenz in ", i, "Schritten zu |Ps-Ps_old| = ",mag(ps - ps_old))
                    break
            ps_old = ps
        if Abbruchkriterium(pe, ps) :
            Integral_gesamt = np.trapezoid(F_Potenzial(ce, pe, cs, ps), dx = h)
            Integral_p = np.trapezoid(F_Potenzial(ce, pe, cs, ps)[L_p:], dx = h)
            print(" Anfangspotenzial: ")
            print(" Konvergenz in ", k,"Schritten")
            print(f"  I_ges: {Integral_gesamt:7.5e}")
            print(f"  Integral_k: {Integral_p:7.5e}")
            break
    return ps, pe

ps_p = np.zeros(length)
# Nach der Initialisierung, füge Debug-Output ein:
print(f"Anfangs-Stöchiometrie Anode: {cs[:L_n].mean() / cmaxn:.3f}")
print(f"Anfangs-Stöchiometrie Kathode: {cs[L_p:].mean() / cmaxp:.3f}")
print(f"Anfangs-OCP Anode: {(Ubsn(cs[:L_n].mean() / cmaxn)/U_ref):.3f} V")
print(f"Anfangs-OCP Kathode: {(Ubsp(cs[L_p:].mean() / cmaxp)/U_ref):.3f} V")

ps, pe = kombistartwert(ps, pe)


# Ausgabe zur Startsituation
ue = pe - np.log(ce) / 2
if info > 3:
   print(f"  Stop zur Zeit: 0")
   print(f"  Ce_min: {min(ce):4.2e} ",f"   Ce_max: {max(ce):4.2e} ")
   print(f"  Cs_min: {min(cs):4.2e} ",f"   Cs_max: {max(cs):4.2e} ")
   print(f"  Ue_min: {min(ue):4.2e} ",f"   Ue_max: {max(ue):4.2e} ")
   print(f"  Pe_min: {min(pe):4.2e} ",f"   Pe_max: {max(pe):4.2e} ")
   print(f"  Ps_min: {min(ps):4.2e} ",f"   Ps_max: {max(ps):4.2e} ")
   Integral_gesamt = np.trapezoid(F_Potenzial(ce, pe, cs, ps),dx = h)
   Integral_p = np.trapezoid(F_Potenzial(ce, pe, cs, ps)[L_p:],dx = h)
   print(" Kontrollausgaben")
   print(f"  I_ges: {Integral_gesamt:7.5e}")
   print(f"  Integral_k: {Integral_p:7.5e}")
if info > 4:
   print(f"  2F*Fluss ce (Rand  links): {2*Faraday*D_e*(ce[ 0] - ce[ 1]) / h:4.2e}")
   print(f"  2F*Fluss ce (Rand rechts): {2*Faraday*D_e*(ce[-1] - ce[-2]) / h:4.2e}")
   print(f"  2F*Fluss cs (Rand  links): {2*Faraday*d_s_n*(cs[ 0] - cs[ 1]) / h:4.2e}")
   print(f"  2F*Fluss cs (Rand rechts): {2*Faraday*d_s_p*(cs[-1] - cs[-2]) / h:4.2e}")
   print(f"  Fluss ue (Rand  links): {kappa_e*(ue[ 0] - ue[ 1]) / h:4.2e}")
   print(f"  Fluss ue (Rand rechts): {kappa_e*(ue[-1] - ue[-2]) / h:4.2e}")
   print(f"  Fluss ps (Rand  links): {kappa_s_n*(ps[ 0] - ps[ 1]) / h:4.2e}")
   print(f"  Fluss ps (Rand rechts): {kappa_s_p*(ps[-1] - ps[-2]) / h:4.2e}")

# ---------------- Haupt-Simulationsschleife ----------------

# Initialisierung
cs_speicher = []
zeit = []

zellspannung = []
# Start-Zellspannung speichern (t = 0)
U_cell_0 = ps[-1] - ps[0]
zellspannung.append(U_cell_0/U_ref)


z = 0
start_time = time.time()
cs_speicher.append(cs.copy())
zeit.append(z) 

# Schleife
for t in range(1,steps+1):
    z += tau    

    # ce berechnen
    b_c = tau * (F_Konzentration(ce, pe, cs, ps) + G_Konzentration)
    ce  = A_c_matrix_lu_zerlegung.solve(ce + b_c)

    # pe berechnen
    pe = meansol(A_u_matrix, f_u, pe)

    # ps berechnen
    ps = berechne_ps(ce, pe, cs, ps)
    U_cell = (ps[-1] - ps[0]) / U_ref

    # cs berechnen
    cs = aktivpartikel(ce, pe, cs, ps)

    if (cs[:L_n].max() > cmaxn or 
        cs[:L_n].min() < 0 or 
        cs[L_p:].max() > cmaxp or 
        cs[L_p:].min() < 0):                            # Falls bereits die maximale/minimale Konzentration erreicht ist
        cs_speicher.append(cs.copy())
        zellspannung.append(U_cell)
        zeit.append(z * T_ref)
        
        print(f"Simulation vorzeitig gestoppt bei {z * T_ref:.0f}s von {T * T_ref:.0f}s")
        break
         
    if t % print_frequency == 0 and info > 0:
        print(f"  Zeit: {(z * T_ref):.0f} s")   # * T_ref sorgt für eine dimensionierte Zeit im Output
        cs_speicher.append(cs.copy())
        zellspannung.append(U_cell)
        zeit.append(z * T_ref)                  # * T_ref sorgt für eine dimensionierte Zeitskala im Diagramm
        print(f"x_n={cs[:L_n].mean()/cmaxn:.3f}, U_n={Ubsn(cs[:L_n].mean()/cmaxn)/U_ref:.3f}V")
        print(f"x_p={cs[L_p:].mean()/cmaxp:.3f}, U_p={Ubsp(cs[L_p:].mean()/cmaxp)/U_ref:.3f}V")
        
# Ende der Schleife

# Aufraeumen
ue = pe - np.log(ce) / 2
end_time = time.time()
 
# Ausgabe
if info > 0:
   print(f"  Stop zur Zeit: {(z * T_ref):.2f}")
   print(f"  Ce_min: {min(ce):4.2e} ",f"   Ce_max: {max(ce):4.2e} ")
   print(f"  Cs_min: {min(cs):4.2e} ",f"   Cs_max: {max(cs):4.2e} ")
   print(f"  Ue_min: {min(ue):4.2e} ",f"   Ue_max: {max(ue):4.2e} ")
   print(f"  Pe_min: {min(pe):4.2e} ",f"   Pe_max: {max(pe):4.2e} ")
   print(f"  Ps_min: {min(ps):4.2e} ",f"   Ps_max: {max(ps):4.2e} ")
   print(f"Benoetigte Rechenzeit: {end_time - start_time:.2f} Sekunden")
   Integral_gesamt = np.trapezoid(F_Potenzial(ce, pe, cs, ps),dx = h)
   Integral_p = np.trapezoid(F_Potenzial(ce, pe, cs, ps)[L_p:],dx = h)
   print(" Kontrollausgaben")
   print(f"  I_ges: {Integral_gesamt:7.5e}")
   print(f"  Integral_k: {Integral_p:7.5e}")
   print(f"  Spannungsdifferenz in der Zelle: {pe[-1]-pe[0]:7.5e}")
if info > 2:
   print(f"  2F*Fluss ce (Rand  links): {2*Faraday*D_e*(ce[ 0] - ce[ 1]) / h:4.2e}")
   print(f"  2F*Fluss ce (Rand rechts): {2*Faraday*D_e*(ce[-1] - ce[-2]) / h:4.2e}")
   print(f"  2F*Fluss cs (Rand  links): {2*Faraday*d_s_n*(cs[ 0] - cs[ 1]) / h:4.2e}")
   print(f"  2F*Fluss cs (Rand rechts): {2*Faraday*d_s_p*(cs[-1] - cs[-2]) / h:4.2e}")
   print(f"  Fluss ue (Rand  links): {kappa_e*(ue[ 0] - ue[ 1]) / h:4.2e}")
   print(f"  Fluss ue (Rand rechts): {kappa_e*(ue[-1] - ue[-2]) / h:4.2e}")
   print(f"  Fluss ps (Rand  links): {kappa_s_n*(ps[ 0] - ps[ 1]) / h:4.2e}")
   print(f"  Fluss ps (Rand rechts): {kappa_s_p*(ps[-1] - ps[-2]) / h:4.2e}")

# ---------------- Visualisierung ----------------
Partikel_werte = matrix_cs[1]
Partikel_0_werte = matrix_cs[0]
Partikel_1_werte = matrix_cs[-1]
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.plot(x, ce, label="Konzentration C_e", color='b')
plt.xlabel("Ort x")
plt.ylabel("Konzentration Elektrolyt")
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(x, pe, label="Potenzial Phi_e", color='g')
plt.xlabel("Ort x")
plt.ylabel("Potenzial Phi Elektrolyt")
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(x, ue, label="Potenzial U Elektrolyt", color='b')
plt.xlabel("Ort x")
plt.ylabel("Potenzial U Elektrolyt")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.plot(x, ps, label="Potenzial_solid Phi_s", color='m')
plt.xlabel("Ort x")
plt.ylabel("Potenzial Solid")
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(x, cs, label="Konzentration_solid C_s", color='m')
plt.xlabel("Ort x")
plt.ylabel("Konzentration C_s")
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(y, Partikel_werte, label="Konzentration_solid Partikel", color='m')
plt.xlabel("Ort x")
plt.ylabel("Konzentration_solid Partikel")
plt.legend()
plt.tight_layout()
plt.show()

#plot Zellspannung
plt.figure(figsize=(8,5))
plt.plot(zeit, zellspannung, linewidth=2)
plt.xlabel("Zeit [s]")
plt.ylabel("Zellspannung [V]")
plt.grid(True)
plt.tight_layout()
plt.show()

if info > 4:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x[:L_n], ps[:L_n], label="Potenzial Phi_s", color='m')
    plt.xlabel("Ort x")
    plt.ylabel("Potenzial Phi Aktivpartikel Anode") 
    plt.legend()
    plt.subplot(1, 2, 2)   
    plt.plot(x[L_p:], ps[L_p:], label="Potenzial Phi_s", color='m')
    plt.xlabel("Ort x")
    plt.ylabel("Potenzial Phi Aktivpartikel Kathode")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(x[:L_n], ps[:L_n], label="Potenzial Phi_s", color='m')
    plt.xlabel("Ort x")
    plt.ylabel("Potenzial Phi Aktivpartikel Anode")

    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(x[L_p:], ps[L_p:], label="Potenzial Phi_s", color='m')
    plt.xlabel("Ort x")
    plt.ylabel("Potenzial Phi Aktivpartikel Kathode")

    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(x, pe, label="Potenzial Phi_e", color='b')
    plt.xlabel("Ort x")
    plt.ylabel("Potenzial Phi Elektrolyt")

    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(x, ue, label="Potenzial U Elektrolyt", color='b')
    plt.xlabel("Ort x")
    plt.ylabel("Potenzial U Elektrolyt")

    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(x, ce, label="Konzentration Elektrolyt", color='b')
    plt.xlabel("Ort x")
    plt.ylabel("Konzentration Elektrolyt")

    plt.show()


colors = plt.cm.viridis(np.linspace(0, 1, max(1, len(cs_speicher))))

for cs_snap, t_snap, c in zip(cs_speicher, zeit, colors):
    plt.plot(x, cs_snap, label=f"t = {t_snap:.2f} s", color=c)

plt.xlabel("Ort x")
plt.ylabel("Konzentration in den Aktivpartikeln C_s")
#plt.title("$C_s$-Profile zu ausgewaehlten Zeiten")
plt.legend(ncols=2)  
plt.tight_layout()
plt.show()


if info > 0:
   print(f"Fertig")
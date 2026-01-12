# database-LGM50.py
# ---------------- Referenzwerte aus Traskunov & Latz (2021) ----------------
# Basiert auf Traskunov & Latz (2021) - Electrochimica Acta 379 (2021) 138144
# LG M50 Graphite open-circuit potential as a function of stoichiometry, fit taken
#     Chang-Hui Chen, Ferran Brosa Planella, Kieran O Regan, Dominika Gastol, W.
#     Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
#     Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
#     Electrochemical Society 167 (2020): 080534.

# Zelltyp: LG M50 (21700 Format, 5 Ah)
# Kathode: NMC (LiNixMnyCozO2)
# Anode: Graphit
# Elektrolyt: LiPF6 in EC:DMC (3:7)

import numpy as np
 
Dimensionierte_Werte = {
    #Universelle Konstanten
    "FarK": 9.64853321e4, #C/mol
    "GasK": 8.3145, #J/(mol·K)
 
    #spezifische Werte
    "L_ges": 1.728e-4, # m - Gesamtdicke (85.2 + 12 + 75.6 µm)
    "L_n": 8.52e-5,    # m - Anoden-Dicke (CHEN2020: "Negative electrode thickness")
    "L_s": 1.2e-5,     # m - Separator-Dicke (CHEN2020: "Separator thickness")  
    "L_p": 7.56e-5,    # m - Kathoden-Dicke (CHEN2020: "Positive electrode thickness")
 
    "R_p": 5.22e-6,    # m - NMC Kathoden-Partikelradius (CHEN2020: "Positive particle radius")
    "R_n": 5.86e-6,    # m - Graphit Anoden-Partikelradius (CHEN2020: "Negative particle radius")

    # Spezifische Oberflächen (berechnet aus Porosität und Partikelradius)
    # a_s = 3 * epsilon_s / R_p  für sphärische Partikel
    # CHEN2020: epsilon_p = 0.665, epsilon_n = 0.75
    "asp": 3 * 0.665 / 5.22e-6,   # 1/m - Spezifische Oberfläche Kathode (~3.82e5 1/m)
    "asn": 3 * 0.75 / 5.86e-6,    # 1/m - Spezifische Oberfläche Anode (~3.84e5 1/m)


    # Zusätzliche Chen2020 Parameter
    "epsilon_p": 0.335,     # Kathoden-Porosität
    "epsilon_n": 0.25,      # Anoden-Porosität
    "epsilon_s": 0.47,      # Separator-Porosität
    "c_max_p": 63104.0,     # mol/m³ - Max. Konzentration Kathode (NMC)
    "c_max_n": 33133.0,     # mol/m³ - Max. Konzentration Anode (Graphit)
    "D_s_p": 1.48e-15,         # m²/s - Festkörper-Diffusivität Kathode -- Experimenteller Wert
    "D_s_n": 1.74e-15,       # m²/s - Festkörper-Diffusivität Anode -- Experimenteller Wert
    "sigma_p": 0.18,        # S/m - Elektronische Leitfähigkeit Kathode
    "sigma_n": 215.0,       # S/m - Elektronische Leitfähigkeit Anode
    "c_e_0": 1000.0,        # mol/m³ - Initiale Elektrolyt-Konzentration
    "t_plus": 0.2594,       # Kationen-Transferzahl'''
 
}
 
def calculate_dimensionless_parameters():
    dim = Dimensionierte_Werte
 
    L_ref = dim["L_ges"]
    C_ref = 64000
    T_ref = 1000
    K_ref = 293 #K
    U_ref= dim["FarK"] / (dim["GasK"] * K_ref)
    J_ref= 1
    
    faraday=dim["FarK"]*L_ref*C_ref/(T_ref*J_ref)
   
    # Dimensionslose Parameter
    parameter = {
        "C_ref": C_ref,
        "T_ref": T_ref,
        "J_ref": J_ref,

        "StoichAt0_p":   0.9084, #Stoichiometry at 0% SOC Positive electrode CHEN2020
        "StoichAt100_p": 0.2661, #Stoichiometry at 100% SOC Positive electrode CHEN2020
        "StoichAt0_n":   0.0279, #Stoichiometry at 0% SOC Negative electrode CHEN2020
        "StoichAt100_n": 0.9014, #Stoichiometry at 100% SOC Negative electrode CHEN2020

        "L_ges": dim["L_ges"],
        "L": dim["L_ges"] / L_ref,
        "L_p": dim["L_p"] / L_ref,
        "L_n": dim["L_n"] / L_ref,
        "L_s": dim["L_s"] / L_ref,  # Separator explizit
        "R_p": dim["R_p"] / L_ref,
        "R_n": dim["R_n"] / L_ref,
        "asn": dim["asn"] * L_ref,
        "asp": dim["asp"] * L_ref,


        "Fara": faraday,
        "U_ref": U_ref,
        "Xi" : faraday/(dim["GasK"]*293),
        
        "kappa_e": 0.12 * U_ref / (L_ref * J_ref),  # Elektrolyt-Leitfähigkeit
        "sigma_n": dim["sigma_n"] * U_ref / (L_ref * J_ref),  # CHEN2020: 215 S/m
        "sigma_p": dim["sigma_p"] * U_ref / (L_ref * J_ref),  # CHEN2020: 0.18 S/m

        "difen": 1e-11 * T_ref / L_ref**2,          # Elektrolyt-Diffusion
        "difsn": dim["D_s_n"] * T_ref / L_ref**2,   # CHEN2020: 3.3e-14 m²/s
        "difsp": dim["D_s_p"] * T_ref / L_ref**2,   # CHEN2020: 4e-15 m²/s

        "ce0": dim["c_e_0"] / C_ref,                # CHEN2020: 1000 mol/m³
        "cmaxn": dim["c_max_n"] / C_ref,            # CHEN2020: 33133 mol/m³
        "cmaxp": dim["c_max_p"] / C_ref,            # CHEN2020: 63104 mol/m³

        "i00": 1/J_ref,
        
        "t_plus": dim["t_plus"],
    }
   
    return parameter
 
 
case_1 = calculate_dimensionless_parameters()

def Ubsp(x):
    """
    Open Circuit Voltage für die Kathode (NMC811) als Funktion der Stöchiometrie.
    Gleichung [8] aus Chen et al. 2020
    """
    U_volt = (
        -0.8090 * x
        + 4.4875
        - 0.0428 * np.tanh(18.5138 * (x - 0.5542))
        - 17.7326 * np.tanh(15.7890 * (x - 0.3117))
        + 17.5842 * np.tanh(15.9308 * (x - 0.3120))
    )

    return U_volt * case_1["U_ref"]
    
def Ubsn(x):
    """
    Open Circuit Voltage für die Anode (Graphit-SiOx) als Funktion der Stöchiometrie.
    Gleichung [9] aus Chen et al. 2020
    """
    U_volt = (
        1.9793 * np.exp(-39.3631 * x)
        + 0.2482 
        - 0.0909 * np.tanh(29.8538 * (x - 0.1234))
        - 0.04478 * np.tanh(14.9159 * (x - 0.2769))
        - 0.0205 * np.tanh(30.4444 * (x - 0.6103))
    )

    return U_volt * case_1["U_ref"]
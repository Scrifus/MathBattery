"""
3D Monte-Carlo-Generator für Ellipsoid-Geometrien
==================================================
 
WISSENSCHAFTLICH FUNDIERTE ASPEKTVERHÄLTNISSE:
 
Die Achsenverhältnisse der Ellipsoide basieren auf gemessenen Werte der Querschnitte der Elektroden.
Die Bilder des Raster-Elektronen-Mikroskops sind in Cheng2020 Figure 3 zu finden (siehe databaseLGM50.py)
Hierbei hat die Kathode annähernd Kugelförmige Aktivpartikel.
Die Anode im Durchschnitt das Verhältniss 3/1. Dahe die Werte:

Anode: Variation_unten=0.7,  Variation_oben=1.5
Kathode: Variation_unten=1.2, Variation_oben=1.2,
"""
 
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, lil_matrix
from databaseLGM50 import case_1 as database
from scipy.sparse import lil_matrix, csc_matrix
import numpy as np
from scipy.optimize import fsolve
from scipy.sparse.linalg import splu
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpmath as mp
import os
 
 
"""
    Parameters:
    Radius : float
        Mittlerer Partikelradius [m]
    Variation_unten : float
        Untere Variation (für oblate/flache Ellipsoide)
    Variation_oben : float
        Obere Variation (für prolate/elongierte Ellipsoide)
    Menge : int
        Anzahl verschiedener Ellipsoidformen                     (Standard = 21)
    Zufallsmenge : int
        Anzahl Monte-Carlo-Samples                               (Standard = 1000)
    electrode_type : str
        'anode' oder 'kathode' für Ausgabedatei                  (Standard = 'anode')
    """
 
def generate_ellipsoid_data(
    Radius,
    Variation_unten,
    Variation_oben,
    Menge=21,
    Zufallsmenge=1000,
    electrode_type='anode'
):
   
    print(f"\n{'='*70}")
    print(f"3D-Ellipsoide für {electrode_type.upper()}")
    print(f"{'='*70}")
    print(f"Radius: {(Radius * 1e6 * database['L_ges']):.2f} µm")
    print(f"Variation_unten: {Variation_unten:.3f} (a/c ≈ {1/Variation_unten:.3f})")
    print(f"Variation_oben: {Variation_oben:.3f} (a/c ≈ {Variation_oben:.3f})")
    print(f"Anzahl Formen: {Menge}")
    print(f"Monte-Carlo Samples: {Zufallsmenge}")
    print(f"{'='*70}\n")
   
    # Halbachsen berechnen
    Halbachse_a = np.linspace(Radius * Variation_oben, (1/Variation_unten) * Radius, Menge)
    Halbachse_c = (Radius**3) / Halbachse_a**2
    Volumen = (4/3) * np.pi * Radius**3
 
    a_spezifisch = []
   
    def Oberfläche_Ellipsoid(a, c):
        """Analytische Oberflächenformel für Rotationsellipsoid."""
        if a > c:  # Oblat (flach)
            return (
                2.0 * math.pi * a**2
                + 2.0 * math.pi * c**2
                  * math.atanh(math.sqrt(1.0 - (c**2) / (a**2)))
                  / math.sqrt(1.0 - (c**2) / (a**2))
            )
        elif c > a:  # Prolat (elongiert)
            return (
                2.0 * math.pi * a**2
                + 2.0 * math.pi * a * c
                  * math.asin(math.sqrt(1.0 - (a**2) / (c**2)))
                  / math.sqrt(1.0 - (a**2) / (c**2))
            )
        elif a == c:  # Kugel
            return 4 * np.pi * a**2
   
    def Ellipsenlage(x, y, z, a, c):
        """Prüft, ob Punkt innerhalb des Ellipsoids liegt."""
        if (x**2/a**2 + y**2/a**2 + z**2/c**2) <= 1:
            return True
        else:
            return False
   
    def b(x, y, z, Basis):
        """10 Basisfunktionen für 3D-Ellipsoide."""
        if Basis == 0:
            return 1
        elif Basis == 1:
            return x
        elif Basis == 2:
            return y
        elif Basis == 3:
            return z
        elif Basis == 4:
            return x**2
        elif Basis == 5:
            return y**2
        elif Basis == 6:
            return z**2
        elif Basis == 7:
            return x * y
        elif Basis == 8:
            return x * z
        elif Basis == 9:
            return y * z
        else:
            raise ValueError("Invalid Basis value")
   
    def grad_b(x, y, z, Basis):
        """Gradient der Basisfunktionen."""
        if Basis == 0:        # 1
            return (0.0, 0.0, 0.0)
        elif Basis == 1:      # x
            return (1.0, 0.0, 0.0)
        elif Basis == 2:      # y
            return (0.0, 1.0, 0.0)
        elif Basis == 3:      # z
            return (0.0, 0.0, 1.0)
        elif Basis == 4:      # x^2
            return (2.0 * x, 0.0, 0.0)
        elif Basis == 5:      # y^2
            return (0.0, 2.0 * y, 0.0)
        elif Basis == 6:      # z^2
            return (0.0, 0.0, 2.0 * z)
        elif Basis == 7:      # x*y
            return (y, x, 0.0)
        elif Basis == 8:      # x*z
            return (z, 0.0, x)
        elif Basis == 9:      # y*z
            return (0.0, z, y)
        else:
            raise ValueError("Invalid Basis value")
   
    def Monte_Carlo_Integration_Ellipse_M(a, c, Basis_1, Basis_2, N):
        """Monte-Carlo-Integration für Massematrix."""
        Treffer = 0
        Integralwert = 0.0
        for i in range(N):
            x = np.random.uniform(-a, a)
            y = np.random.uniform(-a, a)
            z = np.random.uniform(-c, c)
            if Ellipsenlage(x, y, z, a, c):
                Treffer += 1
                Integralwert += b(x, y, z, Basis_1) * b(x, y, z, Basis_2)
        Integralwert = (Integralwert / Treffer) * Volumen
        return Integralwert
   
    def Massematrix(a, c):
        M = lil_matrix((10, 10))
        for i in range(10):
            for j in range(10):
                M[i, j] = Monte_Carlo_Integration_Ellipse_M(a, c, i, j, Zufallsmenge)
        return M
   
    def Steifigkeitsmatrix(a, c):
        """Steifigkeitsmatrix K (10×10)."""
        K = lil_matrix((10, 10))
        for i in range(10):
            for j in range(10):
                def integrand(x, y, z):
                    grad_b_i = grad_b(x, y, z, i)
                    grad_b_j = grad_b(x, y, z, j)
                    return (grad_b_i[0] * grad_b_j[0] +
                            grad_b_i[1] * grad_b_j[1] +
                            grad_b_i[2] * grad_b_j[2])
                Treffer = 0
                Integralwert = 0.0
                for n in range(Zufallsmenge):
                    x = np.random.uniform(-a, a)
                    y = np.random.uniform(-a, a)
                    z = np.random.uniform(-c, c)
                    if Ellipsenlage(x, y, z, a, c):
                        Treffer += 1
                        Integralwert += integrand(x, y, z)
                Integralwert = (Integralwert / Treffer) * Volumen
                K[i, j] = Integralwert
        return K
   
    def Punkt_auf_Rand(a, c):
 
 
        theta = np.random.uniform(0.0, np.pi)       # Polarwinkel
        phi   = np.random.uniform(0.0, 2.0*np.pi)   # Azimut
   
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        cos_p = np.cos(phi)
        sin_p = np.sin(phi)
   
        x = a * sin_t * cos_p
        y = a * sin_t * sin_p
        z = c * cos_t
   
 
        dS_dtheta_dphi = a * sin_t * np.sqrt((c**2) * (sin_t**2) + (a**2) * (cos_t**2))
   
        return x, y, z, dS_dtheta_dphi
   
    def F_Term(a, c, Oberfläche):
        """Randintegral-Vektor F (10,)."""
        F = np.zeros(10, dtype=float)
        for i in range(10):
            Integralwert = 0.0
            Treffer = 0
            for n in range(Zufallsmenge):
                x, y, z, dS_dtheta_dphi = Punkt_auf_Rand(a, c)
                Integralwert += b(x, y, z, i) * dS_dtheta_dphi
                Treffer += 1 * dS_dtheta_dphi
            Integralwert = (Integralwert / Treffer) * Oberfläche
            F[i] = Integralwert
        return F
   
    # ============================================================================
    # HAUPTSCHLEIFE: ALLE ELLIPSOIDFORMEN BERECHNEN
    # ============================================================================
   
    M_liste = []
    K_liste = []
    F_liste = []
    Halbachse_a_liste = []
    Halbachse_c_liste = []
    Umfang_liste = []
   
    start_time = time.time()
   
    for idx in range(Menge):
        print(f"Form {idx+1}/{Menge}: ", end='', flush=True)
        a = Halbachse_a[idx]
        c = Halbachse_c[idx]
        Oberfläche = Oberfläche_Ellipsoid(a, c)
        spezifische_Oberfläche = Oberfläche / Volumen
   
        print(f"a={a*1e6:.3f}µm, c={c*1e6:.3f}µm, a/c={a/c:.3f}", end=' ')
       
        # Matrizen berechnen
        t0 = time.time()
        M = Massematrix(a, c)
        K = Steifigkeitsmatrix(a, c)
        F = F_Term(a, c, Oberfläche)
        dt = time.time() - t0
       
        print(f"[{dt:.1f}s]")
   
        # Daten speichern
        M_liste.append(M.toarray())
        K_liste.append(K.toarray())
        F_liste.append(F)
        a_spezifisch.append(spezifische_Oberfläche)
        Halbachse_a_liste.append(a)
        Halbachse_c_liste.append(c)
        Umfang_liste.append(float(Oberfläche))
   
    elapsed_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Berechnungszeit: {elapsed_time:.1f} Sekunden ({elapsed_time/60:.1f} Minuten)")
    print(f"{'='*70}\n")
   
    # ============================================================================
    # DATEN IN PYTHON-DATEI SPEICHERN
    # ============================================================================
   
    output_file = f'montecarlo_{electrode_type}.py'
   
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Monte Carlo Ergebnisse für {electrode_type.upper()}\n")
        f.write(f"# Radius: {(Radius * 1e6 * database['L_ges']):.3f} µm\n")
        f.write(f"# Variation_unten: {Variation_unten:.3f} (a/c ≈ {1/Variation_unten:.3f})\n")
        f.write(f"# Variation_oben: {Variation_oben:.3f} (a/c ≈ {Variation_oben:.3f})\n")
        f.write(f"# Aspektverhältnis-Bereich: {Halbachse_a[0]/Halbachse_c[0]:.3f} - {Halbachse_a[-1]/Halbachse_c[-1]:.3f}\n")
        f.write(f"# Anzahl Monte-Carlo-Samples: {Zufallsmenge} \n")
        f.write(f"# Anzahl verschiedener Ellipsoidformen: {Menge} \n")
 
        f.write("\nimport numpy as np\n\n")
       
        f.write("Halbachse_a = np.array(")
        f.write(repr(Halbachse_a_liste))
        f.write(")\n\n")
       
        f.write("Halbachse_c = np.array(")
        f.write(repr(Halbachse_c_liste))
        f.write(")\n\n")
   
        f.write("spezifische_Oberflaeche = np.array(")
        f.write(repr(a_spezifisch))
        f.write(")\n\n")
   
        f.write("Umfang = np.array(")
        f.write(repr(Umfang_liste))
        f.write(")\n\n")
   
        f.write("M = np.array([\n")
        for M_array in M_liste:
            f.write("    ")
            f.write(repr(M_array.tolist()))
            f.write(",\n")
        f.write("])\n\n")
       
        f.write("K = np.array([\n")
        for K_array in K_liste:
            f.write("    ")
            f.write(repr(K_array.tolist()))
            f.write(",\n")
        f.write("])\n\n")
       
        f.write("F = np.array([\n")
        for F_array in F_liste:
            f.write("    ")
            f.write(repr(F_array.tolist()))
            f.write(",\n")
        f.write("])\n")
   
    print(f"Daten gespeichert in: {output_file}")
   
    return {
        'M': M_liste,
        'K': K_liste,
        'F': F_liste,
        'Halbachse_a': Halbachse_a_liste,
        'Halbachse_c': Halbachse_c_liste,
        'spezifische_Oberflaeche': a_spezifisch,
        'Umfang': Umfang_liste
    }
 
 
# ============================================================================
# HAUPTPROGRAMM
# ============================================================================
 
if __name__ == '__main__':
    # ANODE: Graphit (flake-shaped)
    print("\n" + "="*70)
    print("ANODE: GRAPHIT (flake-shaped)")
    print("="*70)
    anode_data = generate_ellipsoid_data(
        Radius=database["R_n"],
        Variation_unten=0.7,  
        Variation_oben=1.5,  
        Menge=21,
        Zufallsmenge=1000,
        electrode_type='anode'
    )
   
    # KATHODE: NMC (nearly spherical)
    # Wissenschaftliche Begründung: "nearly spherical" → minimale Abweichung von 1.0
    print("\n" + "="*70)
    print("KATHODE: NMC (nearly spherical)")
    print("="*70)
    kathode_data = generate_ellipsoid_data(
        Radius=database["R_p"],
        Variation_unten=1.2,  
        Variation_oben=1.2,  
        Menge=21,
        Zufallsmenge=1000,
        electrode_type='kathode'
    )
   
    print("\n" + "="*70)
    print("FERTIG!")
    print("="*70)
"""
Multilayer Reflectivity Calculator using xraylib
================================================

Installation:
pip install numpy matplotlib scipy pandas xraylib

Usage:
python multilayer_calculator.py

For Jupyter notebook:
jupyter notebook
# then run the cells

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import pandas as pd

try:
    import xraylib as xrl
    XRAYLIB_AVAILABLE = True
    print("xraylib loaded successfully")
except ImportError:
    XRAYLIB_AVAILABLE = False
    print("WARNING: xraylib not available. Using simplified atomic data.")
    print("Install with: pip install xraylib")

class MultilayerCalculator:
    """
    Professional multilayer reflectivity calculator using xraylib for atomic data.
    """
    
    def __init__(self):
        # Backup atomic data if xraylib not available
        self.backup_atomic_data = {
            'Mo': {'Z': 42, 'A': 95.95, 'density': 10.28},
            'Si': {'Z': 14, 'A': 28.09, 'density': 2.33},
            'Ni': {'Z': 28, 'A': 58.69, 'density': 8.91},
            'C': {'Z': 6, 'A': 12.01, 'density': 2.26},
            'W': {'Z': 74, 'A': 183.84, 'density': 19.25},
            'Cr': {'Z': 24, 'A': 51.996, 'density': 7.19},
            'Sc': {'Z': 21, 'A': 44.956, 'density': 2.99},
            'Ru': {'Z': 44, 'A': 101.07, 'density': 12.37},
            'B': {'Z': 5, 'A': 10.81, 'density': 2.34},
            'O': {'Z': 8, 'A': 15.999, 'density': 1.43}
        }
        
        # Physical constants
        self.r_e = 2.8179403262e-15  # Classical electron radius (m)
        self.hc = 1.23984198e-6      # hc in eV·m
        self.avogadro = 6.02214076e23
        
    def get_atomic_scattering_factors(self, element, energy_kev):
        """
        Get atomic scattering factors using xraylib or backup data.
        
        Parameters:
        -----------
        element : str
            Element symbol (e.g., 'Mo', 'Si')
        energy_kev : float
            Energy in keV
            
        Returns:
        --------
        tuple
            (f1, f2) anomalous scattering factors
        """
        if XRAYLIB_AVAILABLE:
            try:
                # Get atomic number
                if element in ['B4C']:
                    # Handle compounds specially
                    return self._get_compound_scattering('B4C', energy_kev)
                
                Z = xrl.SymbolToAtomicNumber(element)
                f1 = xrl.Fi(Z, energy_kev)
                f2 = xrl.Fii(Z, energy_kev)
                return f1, f2
            except:
                # Fallback to backup data
                return self._get_backup_scattering(element, energy_kev)
        else:
            return self._get_backup_scattering(element, energy_kev)
    
    def _get_backup_scattering(self, element, energy_kev):
        """Simplified scattering factors for when xraylib is not available."""
        if element not in self.backup_atomic_data:
            element = 'Si'  # Default fallback
        
        data = self.backup_atomic_data[element]
        Z = data['Z']
        
        # Very simplified energy dependence
        f1 = Z - 2.0 * np.log(energy_kev + 1)
        f2 = 0.5 * Z / (energy_kev + 1)
        
        return f1, f2
    
    def _get_compound_scattering(self, compound, energy_kev):
        """Handle compound materials like B4C."""
        if compound == 'B4C':
            # B4C: 4 boron + 1 carbon
            f1_B, f2_B = self.get_atomic_scattering_factors('B', energy_kev)
            f1_C, f2_C = self.get_atomic_scattering_factors('C', energy_kev)
            f1 = (4 * f1_B + f1_C) / 5
            f2 = (4 * f2_B + f2_C) / 5
            return f1, f2
        return 6.0, 0.1  # Fallback
    
    def get_density(self, material):
        """Get material density."""
        densities = {
            'Mo': 10.28, 'Si': 2.33, 'Ni': 8.91, 'C': 2.26, 'W': 19.25,
            'Cr': 7.19, 'Sc': 2.99, 'Ru': 12.37, 'B4C': 2.52,
            'SiO2': 2.2, 'Si3N4': 3.2, 'Pt': 21.45
        }
        return densities.get(material, 2.33)
    
    def get_atomic_mass(self, material):
        """Get atomic mass."""
        masses = {
            'Mo': 95.95, 'Si': 28.09, 'Ni': 58.69, 'C': 12.01, 'W': 183.84,
            'Cr': 51.996, 'Sc': 44.956, 'Ru': 101.07, 'B4C': 55.25,
            'SiO2': 60.08, 'Si3N4': 140.28, 'Pt': 195.08
        }
        return masses.get(material, 28.09)
    
    def complex_refractive_index(self, material, energy_ev, custom_density=None):
        """
        Calculate complex refractive index.
        
        Parameters:
        -----------
        material : str
            Material name
        energy_ev : float
            Energy in eV
        custom_density : float, optional
            Custom density in g/cm³
            
        Returns:
        --------
        complex
            Complex refractive index n = 1 - delta - i*beta
        """
        energy_kev = energy_ev / 1000.0
        f1, f2 = self.get_atomic_scattering_factors(material, energy_kev)
        
        density = custom_density if custom_density else self.get_density(material)
        atomic_mass = self.get_atomic_mass(material)
        
        # Number density (atoms/m³)
        n_atoms = density * 1e3 * self.avogadro / (atomic_mass * 1.66054e-27)
        
        # Calculate delta and beta
        wavelength = self.hc / energy_ev
        prefactor = self.r_e * n_atoms * wavelength**2 / (2 * np.pi)
        
        delta = prefactor * f1
        beta = prefactor * f2
        
        return 1 - delta - 1j * beta
    
    def calculate_reflectivity_parratt(self, layers, energies, angle_deg=5.0):
        """
        Calculate reflectivity using Parratt recursion formula.
        
        Parameters:
        -----------
        layers : list of dict
            Layer structure: [{'material': str, 'thickness': float (nm), 
                              'roughness': float (nm), 'density': float (optional)}]
        energies : array-like
            Energies in eV
        angle_deg : float
            Grazing angle in degrees
            
        Returns:
        --------
        array
            Reflectivity values
        """
        angle_rad = np.deg2rad(angle_deg)
        reflectivities = []
        
        for energy in energies:
            wavelength = self.hc / energy
            k0 = 2 * np.pi / wavelength
            
            # Get refractive indices
            n_layers = []
            for layer in layers:
                density = layer.get('density', None)
                n = self.complex_refractive_index(layer['material'], energy, density)
                n_layers.append(n)
            
            # Add substrate (use last layer material)
            n_substrate = n_layers[-1] if n_layers else 1.0
            
            # Parratt recursion from bottom to top
            # Start with no reflection from infinite substrate
            r = 0.0
            
            # Work backwards through layers
            for i in range(len(layers) - 1, -1, -1):
                n_j = n_layers[i]
                thickness = layers[i]['thickness'] * 1e-9  # Convert to meters
                roughness = layers[i]['roughness'] * 1e-9
                
                # Next medium (substrate or next layer)
                if i == len(layers) - 1:
                    n_next = n_substrate
                else:
                    n_next = n_layers[i + 1]
                
                # Calculate wave vector components
                sin_alpha = np.sin(angle_rad)
                
                # Handle complex square roots properly
                cos_alpha_j = np.sqrt(1 - (sin_alpha / n_j)**2 + 0j)
                cos_alpha_next = np.sqrt(1 - (sin_alpha / n_next)**2 + 0j)
                
                # Fresnel coefficient at interface
                r_fresnel = (n_j * cos_alpha_j - n_next * cos_alpha_next) / \
                           (n_j * cos_alpha_j + n_next * cos_alpha_next)
                
                # Roughness damping (Névot-Croce)
                if roughness > 0:
                    sigma_eff = roughness * 2 * k0 * np.sqrt(
                        np.real(n_j * cos_alpha_j * np.conj(n_next * cos_alpha_next))
                    )
                    roughness_factor = np.exp(-sigma_eff**2)
                    r_fresnel *= roughness_factor
                
                # Phase factor
                beta = 2j * k0 * n_j * cos_alpha_j * thickness
                
                # Parratt recursion
                numerator = r_fresnel + r * np.exp(beta)
                denominator = 1 + r_fresnel * r * np.exp(beta)
                
                # Avoid division by zero
                if np.abs(denominator) > 1e-10:
                    r = numerator / denominator
                else:
                    r = r_fresnel
            
            # Final interface with vacuum
            if len(layers) > 0:
                n_top = n_layers[0]
                cos_alpha_0 = np.cos(angle_rad)
                cos_alpha_top = np.sqrt(1 - (np.sin(angle_rad) / n_top)**2 + 0j)
                
                r_top = (cos_alpha_0 - n_top * cos_alpha_top) / \
                        (cos_alpha_0 + n_top * cos_alpha_top)
                
                # Apply top interface roughness
                if layers[0]['roughness'] > 0:
                    sigma_top = layers[0]['roughness'] * 1e-9 * 2 * k0 * \
                               np.sqrt(np.real(cos_alpha_0 * np.conj(n_top * cos_alpha_top)))
                    r_top *= np.exp(-sigma_top**2)
                
                # Final combination
                thickness_top = layers[0]['thickness'] * 1e-9
                beta_top = 2j * k0 * n_top * cos_alpha_top * thickness_top
                
                numerator = r_top + r * np.exp(beta_top)
                denominator = 1 + r_top * r * np.exp(beta_top)
                
                if np.abs(denominator) > 1e-10:
                    r_total = numerator / denominator
                else:
                    r_total = r_top
            else:
                r_total = 0.0
            
            reflectivity = np.abs(r_total)**2
            reflectivities.append(np.real(reflectivity))
        
        return np.array(reflectivities)
    
    def create_multilayer(self, materials, thicknesses, n_bilayers, 
                         roughnesses=None, densities=None):
        """
        Create multilayer structure.
        
        Parameters:
        -----------
        materials : list
            List of materials in one period
        thicknesses : list
            Thicknesses in nm
        n_bilayers : int
            Number of repetitions
        roughnesses : list, optional
            Interface roughnesses in nm
        densities : list, optional
            Custom densities in g/cm³
            
        Returns:
        --------
        list
            Layer structure
        """
        if roughnesses is None:
            roughnesses = [0.3] * len(materials)
        if densities is None:
            densities = [None] * len(materials)
        
        layers = []
        for bilayer in range(n_bilayers):
            for i, material in enumerate(materials):
                layer = {
                    'material': material,
                    'thickness': thicknesses[i],
                    'roughness': roughnesses[i]
                }
                if densities[i] is not None:
                    layer['density'] = densities[i]
                layers.append(layer)
        
        return layers
    
    def optimize_thickness(self, materials, energy_ev, n_bilayers=50, 
                          thickness_range=(2, 15), angle_deg=5.0):
        """
        Optimize total bilayer thickness for maximum reflectivity.
        """
        def objective(total_thickness):
            # Distribute thickness (40% absorber, 60% spacer for typical systems)
            if len(materials) == 2:
                t1 = total_thickness * 0.4
                t2 = total_thickness * 0.6
                thicknesses = [t1, t2]
            else:
                # Equal distribution for other cases
                thicknesses = [total_thickness / len(materials)] * len(materials)
            
            layers = self.create_multilayer(materials, thicknesses, n_bilayers)
            reflectivity = self.calculate_reflectivity_parratt(layers, [energy_ev], angle_deg)[0]
            return -reflectivity
        
        result = minimize_scalar(objective, bounds=thickness_range, method='bounded')
        
        optimal_total = result.x
        max_reflectivity = -result.fun
        
        if len(materials) == 2:
            optimal_thicknesses = [optimal_total * 0.4, optimal_total * 0.6]
        else:
            optimal_thicknesses = [optimal_total / len(materials)] * len(materials)
        
        return {
            'optimal_total_thickness': optimal_total,
            'optimal_thicknesses': optimal_thicknesses,
            'max_reflectivity': max_reflectivity,
            'success': result.success
        }

def plot_multilayer_results(energies, reflectivities, title="Multilayer Reflectivity",
                          angle=5.0, save_file=None):
    """Create professional plots."""
    plt.figure(figsize=(12, 8))
    
    # Main plot
    plt.subplot(2, 1, 1)
    plt.semilogy(energies, reflectivities, 'b-', linewidth=2.5, label='Reflectivity')
    plt.xlabel('Energy (eV)', fontsize=12, fontweight='bold')
    plt.ylabel('Reflectivity', fontsize=12, fontweight='bold')
    plt.title(f'{title} (θ = {angle}°)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.4)
    plt.legend()
    
    # Find and annotate peak
    max_idx = np.argmax(reflectivities)
    max_energy = energies[max_idx]
    max_refl = reflectivities[max_idx]
    plt.annotate(f'Peak: {max_refl:.1%}\nat {max_energy:.1f} eV',
                xy=(max_energy, max_refl), 
                xytext=(max_energy + (energies[-1] - energies[0]) * 0.1, max_refl * 0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, color='red', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Linear scale subplot
    plt.subplot(2, 1, 2)
    plt.plot(energies, reflectivities * 100, 'g-', linewidth=2.5)
    plt.xlabel('Energy (eV)', fontsize=12, fontweight='bold')
    plt.ylabel('Reflectivity (%)', fontsize=12, fontweight='bold')
    plt.title('Linear Scale', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.4)
    plt.ylim(0, max(reflectivities) * 105)
    
    plt.tight_layout()
    
    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    
    plt.show()

def main():
    """Main demonstration script."""
    print("Multilayer Reflectivity Calculator with xraylib")
    print("=" * 50)
    
    if not XRAYLIB_AVAILABLE:
        print("Note: Running with simplified atomic data (xraylib not available)")
        print("For accurate results, install xraylib: pip install xraylib\n")
    
    calc = MultilayerCalculator()
    
    # Example 1: Mo/Si multilayer
    print("1. Mo/Si Multilayer Analysis")
    print("-" * 30)
    
    mo_si_layers = calc.create_multilayer(
        materials=['Mo', 'Si'],
        thicknesses=[2.8, 4.2],  # nm
        n_bilayers=50,
        roughnesses=[0.3, 0.3]
    )
    
    energies = np.linspace(85, 95, 100)
    reflectivities = calc.calculate_reflectivity_parratt(mo_si_layers, energies, angle_deg=5.0)
    
    max_refl = np.max(reflectivities)
    max_energy = energies[np.argmax(reflectivities)]
    
    print(f"Peak reflectivity: {max_refl:.1%} at {max_energy:.1f} eV")
    print(f"Total thickness: {50 * (2.8 + 4.2):.0f} nm")
    
    plot_multilayer_results(energies, reflectivities, "Mo/Si Multilayer", 5.0)
    
    # Example 2: Optimization
    print("\n2. Thickness Optimization")
    print("-" * 25)
    
    optimization = calc.optimize_thickness(['Mo', 'Si'], energy_ev=92.0, n_bilayers=50)
    
    print(f"Optimal Mo thickness: {optimization['optimal_thicknesses'][0]:.2f} nm")
    print(f"Optimal Si thickness: {optimization['optimal_thicknesses'][1]:.2f} nm")
    print(f"Total bilayer: {optimization['optimal_total_thickness']:.2f} nm")
    print(f"Maximum reflectivity: {optimization['max_reflectivity']:.1%}")
    
    # Example 3: System comparison
    print("\n3. Multiple System Comparison")
    print("-" * 30)
    
    systems = {
        'Mo/Si (EUV)': {
            'materials': ['Mo', 'Si'],
            'thicknesses': [2.8, 4.2],
            'energy_range': (85, 95),
            'angle': 5.0
        },
        'Ni/C (Soft X-ray)': {
            'materials': ['Ni', 'C'],
            'thicknesses': [1.5, 3.5],
            'energy_range': (260, 290),
            'angle': 1.0
        },
        'W/Si (Hard X-ray)': {
            'materials': ['W', 'Si'],
            'thicknesses': [1.8, 2.2],
            'energy_range': (8000, 9000),
            'angle': 0.5
        }
    }
    
    plt.figure(figsize=(15, 5))
    
    for i, (name, system) in enumerate(systems.items()):
        layers = calc.create_multilayer(
            materials=system['materials'],
            thicknesses=system['thicknesses'],
            n_bilayers=50
        )
        
        e_min, e_max = system['energy_range']
        energies = np.linspace(e_min, e_max, 100)
        reflectivities = calc.calculate_reflectivity_parratt(
            layers, energies, angle_deg=system['angle']
        )
        
        plt.subplot(1, 3, i+1)
        plt.semilogy(energies, reflectivities, linewidth=2.5)
        plt.xlabel('Energy (eV)')
        plt.ylabel('Reflectivity')
        plt.title(f'{name}')
        plt.grid(True, alpha=0.4)
        
        max_refl = np.max(reflectivities)
        print(f"{name}: Peak = {max_refl:.1%}")
    
    plt.tight_layout()
    plt.show()
    
    # Export data
    df = pd.DataFrame({
        'Energy_eV': energies,
        'Reflectivity': reflectivities,
        'Reflectivity_Percent': reflectivities * 100
    })
    df.to_csv('multilayer_results.csv', index=False)
    print(f"\nData exported to multilayer_results.csv")
    print("Calculation complete!")

if __name__ == "__main__":
    main()
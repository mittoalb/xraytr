# X-ray server Tools

A dual-purpose web application for X-ray absorption spectroscopy analysis and reference data lookup.

## Overview

This application provides two main functionalities:
1. X-ray transmissivity and refractive index calculations for materials
2. Comprehensive X-ray absorption edge reference table

## Requirements

### Python Dependencies
- dash
- plotly
- pandas
- numpy
- requests
- xraylib

### Installation
```bash
pip install dash plotly pandas numpy requests xraylib
```

## Files

- `main_app.py` - Main Dash application
- `xray_edge_data.py` - X-ray absorption edge database
- `densities.csv` - Optional density cache file (auto-generated)

## Usage

### Running the Application
```bash
python main_app.py
```

The application will start on `http://localhost:8009`

### Tab 1: Transmissivity & Refractive Index Calculator

Calculate X-ray transmission properties for materials:

**Inputs:**
- Formula: Chemical formula (e.g., SiO2, Fe, CaCO3)
- Density: Material density in g/cm³ (auto-detected when possible)
- Thickness: Sample thickness in mm
- Energy: X-ray energy in keV (single value or range as start:stop:step)

**Outputs:**
- Transmissivity plot vs energy
- Refractive index components (delta and beta) vs energy

**Density Resolution:**
The application attempts to resolve material densities in the following order:
1. User-provided value
2. PubChem database lookup (for compounds)
3. XRayLib elemental density (for elements)
4. Local cache file
5. Default value (1.0 g/cm³)

### Tab 2: X-ray Absorption Edge Reference

Interactive table of X-ray absorption edges for all elements:

**Features:**
- Search by element name or symbol
- Filter by edge type (K, L₁, L₂, L₃, or all)
- Energy range filtering
- Sortable columns
- Built-in table search and pagination

**Data Coverage:**
- Elements Z=1 to Z=103
- K-edge energies for all elements
- L-edge energies (L₁, L₂, L₃) where available
- All energies in eV

## Technical Details

### XRayLib Integration
The transmissivity calculations use the XRayLib library for:
- Total cross-sections
- Refractive index calculations
- Elemental densities

### Data Sources
X-ray edge energies compiled from standard spectroscopic references. The database includes absorption edges commonly used in X-ray absorption spectroscopy.

### Performance Notes
- Density lookups from PubChem may introduce network delays
- Results are cached locally to improve subsequent performance
- Large energy ranges may require processing time

## File Structure

```
project/
├── main_app.py           # Main application
├── xray_edge_data.py     # Edge database
├── densities.csv         # Auto-generated cache
└── README.md             # This file
```

## Limitations

- Network connection required for PubChem density lookups
- XRayLib must be properly installed and initialized
- Some L-edge data may not be available for lighter elements
- Energy calculations limited by XRayLib supported range

## Troubleshooting

**XRayLib Errors:**
Ensure XRayLib is properly installed and can access its data files.

**Network Timeouts:**
PubChem lookups have a 5-second timeout. Local density values will be used as fallback.

**Callback Exceptions:**
The application uses dynamic tab rendering. Callback exceptions are suppressed to handle component loading.

## Output Data

The application generates:
- Interactive plots in PNG/SVG format (via Plotly)
- Numerical data accessible through plot interfaces
- CSV cache file for density data persistence

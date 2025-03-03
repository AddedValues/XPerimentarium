"""
Computes performance metrics for a heat pump using CoolProp.

The performance metrics are computed relative to the compression volume flow rate (swallowing ability).
The performance metrics are:
- Work: Compression work.
- qEvap: Latent heat absorbed.
- qOut: Heat output.
- Yield: Heat output normed by yield at source temperature of 4.0°C and drain temperature of 40.0°C.
- COP: Coefficient of performance.

CoolProp is a C++ library that implements the thermophysical properties of fluids.
See: http://www.coolprop.org/index.html 

conda install -c conda-forge::coolprop
pip install pyfluids

Author: Mogens Bech Laursen
2025 March 03
"""

import os
import numpy as np
import pandas as pd
# from pyfluids import Fluid, FluidsList
from CoolProp.CoolProp import PropsSI
import matplotlib.pyplot as plt
import xlwings as xw

os.chdir(os.path.join(os.getcwd(), 'CoolProp'))
print(f'curdir: {os.getcwd()}')

# Reference state for CoolProp shall be set to h=200 kJ/kg and s=1 kJ/kgK for saturated liquid at 0°C. (IIR).
global TKelvin
TKelvin = 273.15
refrigerants = ['NH3', 'CO2']
hOfzRef = {refrig: PropsSI('H', 'T', 0.0+TKelvin, 'Q', 0, refrig) - 200E3 for refrig in refrigerants}
sOfzRef = {refrig: PropsSI('S', 'T', 0.0+TKelvin, 'Q', 0, 'NH3') - 1E3 for refrig in refrigerants}
print(f'hOfz: {hOfzRef}')
print(f'sOfz: {sOfzRef}')

# def is_file_synchronized(file_path: str) -> bool:
#     """
#     Check if the file is synchronized to the local machine using Dropbox.
#     """
#     dbx = dropbox.Dropbox('YOUR_ACCESS_TOKEN')

#     try:
#         metadata = dbx.files_get_metadata(file_path)
#         return metadata.server_modified is not None
#     except dropbox.exceptions.ApiError as e:
#         print(f"API error: {e}")
#         return False

def calc(TsourceC: float, TdrainC: float, pCond: float, refrig: str) -> list[float]:
    """ \
    Calculate the performance metrics of a heat pump:
    - TsourceC: Source temperature in Celsius.
    - TdrainC: Drain temperature in Celsius.
    - TcondC: Condenser temperature in Celsius.
    - refrig: Refrigerant name.
    Returns:
    - Tsource, Tdrain, Tcond (input temperatures)
    - pA: Pressure at source temperature and quality 0 (liquid).
    - hA: Enthalpy at source temperature and quality 0 (liquid).
    - hB: Enthalpy at source temperature and quality 1 (saturated vapour).
    - hC: Enthalpy at condenser pressure and entropy at B.
    - hD: Enthalpy at drain temperature and quality 0 (liquid).
    - sB: Entropy at source temperature and quality 1 (saturated vapour).
    - work: Compression work.
    - qEvap: Latent heat absorbed.
    - qOut: Heat output.
    - yield: Heat output.
    - COP: Coefficient of performance.
    The performance metrics are computed relative to the compression volume flow rate (swallowing ability).

    """
    Tsource = TsourceC + TKelvin
    Tdrain  = TdrainC  + TKelvin
    pA    = PropsSI('P', 'T', Tsource, 'Q', 0,     refrig)    # Pressure at source temperature and quality 0 (liquid).
    hD    = PropsSI('H', 'T', Tdrain,  'P', pCond, refrig)    # Enthalpy at drain temperature and quality 0 (liquid).
    hA    = hD
    hB    = PropsSI('H', 'T', Tsource, 'Q', 1,     refrig)    # Enthalpy at source temperature and quality 1 (saturated vapour).
    dhAB  = hB - hA
    densB = PropsSI('D', 'T', Tsource, 'Q', 1, refrig)        # Density at source temperature and quality 1 (saturated vapour).
    sB    = PropsSI('S', 'T', Tsource, 'Q', 1, refrig)        # Entropy at source temperature and quality 1 (saturated vapour).
    hC    = PropsSI('H', 'P', pCond,   'S', sB, refrig)       # Enthalpy at condenser pressure and entropy at B.

    # Performance metrics relative to the compression volume flow rate (swallowing ability).
    work  = densB * (hC - hB)     # Compression work.
    qEvap = densB * dhAB          # Latent heat absorbed.
    y     = work + qEvap          # Heat output.
    COP   = y / work              # Coefficient of performance.

    hOfz = hOfzRef[refrig]
    sOfz = sOfzRef[refrig]
    result = {'Tsource': TsourceC, 'Tdrain': TdrainC, 'pCond': pCond, \
            'pA': pA, 'hA': hA-hOfz, 'hB': hB-hOfz, 'hC': hC-hOfz, 'hD': hD-hOfz, 'sB': sB-sOfz, \
            'Work': work, 'qEvap': qEvap, 'qOut': work+qEvap, 'Yield': y, 'COP': COP}
    return result

TcondsC = [60.0, 70.0, 80.0]     # Forward condenser temperature in Celsius for NH3.
pConds  = [90E5, 100E5, 110E5]   # Forward condenser pressure in Pa for CO2.
TsourceRefC =  4.0               # Independent of refrigerant, tied to source inlet temperature.
TdrainRefC  = 40.0               # Independent of refrigerant, tied to district heating return temperature.

results: dict[str, pd.DataFrame] = {}  # Key is combination of refrigerant and condenser state (temp. or pressure).

for refrig in refrigerants:
    for iCond in range(3):  # Loops either TcondsC or Pconds

        dfResults = pd.DataFrame(columns=['Tsource', 'Tdrain', 'pCond', 'pA', 'hA', 'hB', 'hC', 'hD', 'sB', 'Work', 'qEvap', 'qOut', 'Yield', 'COP'])

        TdrainsC  = np.arange(30.0, 50.0+0.5, 2.0) 
        TsourcesC = np.arange(00.0, 20.0+0.5, 2.0)  
        if refrig == 'NH3':
            TcondC  = TcondsC[iCond]       # Forward condenser temperature in Celsius.
            Tcond   = TcondC   + TKelvin   # Underkritisk NH3.
            title = f'{refrig} at Tcond={int(TcondC)}°C'
            pCond = PropsSI('P', 'T', Tcond, 'Q', 0, refrig)    # Condensation pressure at condensor source temperature and quality 0 (liquid).
        elif refrig == 'CO2':
            pCond = pConds[iCond]    # Forward condenser pressure in bar for CO2.
            title = f'{refrig} at pCond={int(pCond/1E5)} bar'

        try:
            for TsourceC in TsourcesC:
                for TdrainC in TdrainsC:
                    dfResults.loc[len(dfResults), :] = calc(TsourceC, TdrainC, pCond, refrig)
        
        except Exception as e:
            print(f'Error: {e}')
            print(f'TsourceC: {TsourceC}, TdrainC: {TdrainC}, pCond: {pCond}, refrig: {refrig}')

        # Compute the performance metrics at reference state and normalize the yield factor.
        resultsRef = calc(TsourceRefC, TdrainRefC, pCond, refrig)
        dfResults['YieldNormed'] = dfResults['Yield'] / resultsRef['Yield']

        # Create pivot tables for the performance metrics.
        dfYield = dfResults.pivot(index='Tsource', columns='Tdrain', values='YieldNormed')
        dfCOP   = dfResults.pivot(index='Tsource', columns='Tdrain', values='COP')

        results[title] = (dfResults, dfYield, dfCOP)

# Write the results to Excel files.
xlApp = xw.App(visible=True, add_book=True)
wb = xlApp.books.active
for title in results.keys():
    dfResults, dfYield, dfCOP = results[title]   # Unpacking.
    sh = wb.sheets.add(name=f'dfResults_{title}')
    sh.range('A1').value = title
    sh.range('A2').value = 'All results'
    sh.range('A10').value = dfResults

    sh = wb.sheets.add(name=f'dfYield_{title}')
    sh.range('A1').value = title
    sh.range('A2').value = 'Yield'
    sh.range('A10').value = dfYield

    sh = wb.sheets.add(name=f'dfCOP_{title}')
    sh.range('A1').value = title
    sh.range('A2').value = 'Carnot COP'
    sh.range('A10').value = dfCOP

wb.save('HP performance metrics.xlsx')
wb.close()
xlApp.quit()

# conda install -c conda-forge::coolprop
# pip install pyfluids

import os
import sys
import time
import numpy as np
# import pandas as pd
from pyfluids import Fluid, FluidsList
from CoolProp.CoolProp import PropsSI

water_vapour = Fluid(FluidsList.Water).dew_point_at_pressure(101325)
# print(water_vapour.specific_heat)  # 2079.937085633241

nh3 = Fluid(FluidsList.Ammonia)
co2 = Fluid(FluidsList.CarbonDioxide)

rho = PropsSI('D', 'T', 273.15, 'Q', 0, 'NH3')  # 817.0
# print(rho)

# refrig = 'CO2'
refrig = 'NH3'
TsourceRefs = {'NH3': 40.0, 'CO2': 30.0}
PliquidRefs = {'NH3': 40E5, 'CO2': 70E5}
T0 = 273.15
Tsource = 7.0 + T0 
TsourceRef = TsourceRefs[refrig] + T0  # Reference drain temperature for computing yield sensitivity to drain temperature.
Tdrains = np.arange(30.0, 70.0+0.5, 10.0) + T0
PliquidRef = PliquidRefs[refrig]

hLiqSatRef = PropsSI('H', 'T', TsourceRef, 'Q', 0, refrig)
hVapSatRef = PropsSI('H', 'T', TsourceRef, 'Q', 1, refrig)
dhRef = hVapSatRef - hLiqSatRef

print(f'TsourceRef={TsourceRef-T0:.0f}: {hLiqSatRef=:.0f}, {hVapSatRef=:.0f}, {dhRef=:.0f}')

for Tdrain in Tdrains:
    hLiq = PropsSI('H', 'T', Tdrain, 'P', PliquidRef, refrig)
    dH = hVapSatRef - hLiq
    dYield = dH / dhRef
    print(f'Tdrain={Tdrain-T0}: {dH=:.0f}, {dYield=:.3f}')

    pass

pass
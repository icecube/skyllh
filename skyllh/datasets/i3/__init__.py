from skyllh.datasets.i3 import (
    PublicData_10y_ps,
    PublicData_10y_ps_wMC,
    PublicData_14y_ps,
    TestData,
)

data_samples = {
    'IceTracks-DR1': PublicData_10y_ps,
    'IceTracks-DR2': PublicData_14y_ps,
    # For internal use:
    'IceTracks-DR1_wMC': PublicData_10y_ps_wMC,
    'TestData': TestData,
}

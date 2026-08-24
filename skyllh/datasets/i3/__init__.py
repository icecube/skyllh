from skyllh.datasets.i3 import (
    PublicData_10y_ps,
    PublicData_10y_ps_wMC,
    PublicData_14y_ps,
    PublicData_14y_ps_v2,
    TestData,
)

data_samples = {
    'IceTracks-DR1': PublicData_10y_ps,
    'IceTracks-DR2': PublicData_14y_ps_v2,
    'IceTracks-DR2-v1': PublicData_14y_ps,
    'IceTracks-DR2-v2': PublicData_14y_ps_v2,
    # For internal use:
    'IceTracks-DR1_wMC': PublicData_10y_ps_wMC,
    'TestData': TestData,
}

#!/bin/bash

mkdir -p /data/user/tkontrimas/datarelease_2025/icecube_pstracks_v004p02

ln -s /data/user/wluszczak/datarelease_2025/wg-nu-sources/wg-scripts/data-release/pstracks_v004p02/icecube_pstracks_v004p02/events /data/user/tkontrimas/datarelease_2025/icecube_pstracks_v004p02/
ln -s /data/user/wluszczak/datarelease_2025/wg-nu-sources/wg-scripts/data-release/pstracks_v004p02/icecube_pstracks_v004p02/uptime /data/user/tkontrimas/datarelease_2025/icecube_pstracks_v004p02/

ln -s /data/user/wluszczak/datarelease_2025/wg-nu-sources/wg-scripts/data-release/pstracks_v004p02/icecube_pstracks_v004p02/irfs/*.csv /data/user/tkontrimas/datarelease_2025/icecube_pstracks_v004p02/irfs/

ln -s /data/user/wluszczak/datarelease_2025/bintest/icecube_pstracks_v004p02/irfs/* /data/user/tkontrimas/datarelease_2025/icecube_pstracks_v004p02/irfs/

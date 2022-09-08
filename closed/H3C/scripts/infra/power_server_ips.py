# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__doc__ = """Script used to store, retrieve, and check connectivity to the IP addresses of Power Meter servers."""

import argparse
import os
import subprocess
import sys

POWER_SERVER_IP_MAP = {
    "h3cserver": "172.16.36.116",  
    "ro-dvt-060-80gb": "10.117.19.42",  # ipp1-1644
    "ro-dvt-053-80gb": "10.117.19.42",  # ipp1-1644
    "altra-g242-p31-01": "10.117.20.27",  # ipp1-2564
    "ipp1-1469": "10.117.17.76",  # ipp1-2563
    "sjc1-luna-02": "10.117.17.74",  # ipp1-2422
    "ipp1-2468-jetson": "10.117.21.68",  # ipp1-2423
    "ipp1-2469-jetson": "10.117.22.159",  # ipp1-1697
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sut_hostname",
        help="Hostname of the SUT that power is measured for"
    )
    parser.add_argument(
        "--check_connectivity",
        help="Check connectivity status and SSH login validity, instead of printing the IP address.",
        action="store_true"
    )
    return parser.parse_args()


def check_connectivity(ip_addr):
    ssh_command = f"ping -c 1 {ip_addr}"
    return subprocess.call(ssh_command, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, shell=True)


if __name__ == "__main__":
    args = get_args()
    if args.sut_hostname not in POWER_SERVER_IP_MAP:
        sys.exit(1)

    ip_addr = POWER_SERVER_IP_MAP[args.sut_hostname]
    if args.check_connectivity:
        sys.exit(check_connectivity(ip_addr))
    else:
        print(ip_addr)

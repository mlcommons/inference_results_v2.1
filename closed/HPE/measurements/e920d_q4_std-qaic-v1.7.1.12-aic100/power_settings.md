# BIOS Firmware Settings

## Performance Tuning Options

`Performance > Workload Performance Advisor > Performance Tuning Options`

### Sub-NUMA clustering: Disabled
### NUMA Group Size Optimization: Flat
### Uncore Frequency Scaling: Auto
### Memory Refresh Rate: 1x Refresh
### Power Regulator: OS Control Mode
### Minimum Processor Idle Power Package C-State: Package C6 (non-retention) State
### Energy/Performance Bias: Power Savings Mode

# Management Firmware Settings

## iLo 5 (2.70 May 16 2022)

Out-of-the-box.

# Power Management Settings

## Fans

`Power & Thermal > Fans`

### Minimum Fan Speed: 0%
### Thermal Configuration: Increased Cooling

## Power Regulator Settings

`Power & Thermal > Power Settings > Power Regulator Settings`

### Power Regulator: OS Control Mode

## Maximum Frequency

The maximum chip frequency is controlled through a variable called `vc`.
This variable is set automatically per workload and per system according
to [cmdgen metadata](https://github.com/krai/ck-qaic/tree/main/cmdgen).

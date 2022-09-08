# BIOS Firmware Settings

## System Profile Settings

`Configuration > BIOS Settings > System Profile Settings`

### System Profile: Custom
### CPU Power Management: OS DBPM
### Memory Frequency: Maximum Performance
### Turbo Boost: Enabled
### C States: Disabled
### Write Data CRC: Disabled
### Memory Patrol Scrub: Standard
### Memory Refresh Rate: 1x
### Workload Profile: Not Configured
### PCI ASPM L1 Link Power Management: Enabled
### Determinism Slider: Power Determinism
### Efficiency Optimized Mode: Enabled
### Algorithm Performance Boost Disable (ApbDis): Disabled

## Processor Settings

`Configuration > BIOS Settings > Processor Settings`

### Logical Processor: Enabled
### Virtualization Technology: Enabled
### IOMMU Support: Enabled
### Kernel DMA Protection: Disabled
### L1 Stream HW Prefetcher: Enabled
### L2 Stream HW Prefetcher: Enabled
### MADT Core Enumeration: Linear
### NUMA Nodes Per Socket: 1
### L3 cache as NUMA Domain: Disabled
### Minimum SEV non-ES ASID: 1
### Transparent Secure Memory Encryption: Disabled
### Configurable TDP: Minimum
### x2APIC Mode: Enabled
### Number of CCDs per Processor: All
### Number of Cores per CCD: All

# Management Firmware Settings

## Integrated Dell Remote Access Controller 9

Out-of-the-box.

# Power Management Settings

## Cooling Configuration

`Configuration > System Settings > Hardware Settings > Cooling Configuration`

### Automatic Fan Speed Calculation
#### Thermal Profile Optimization: Minimum Power (Performance per Watt Optimized)

### Fan Speed Offset
#### Fan Speed Offset: Off

### Thresholds
#### Minimum Fan Speed in PWM (% of Max): Default

## Maximum Frequency

The maximum chip frequency is controlled through a variable called `vc`.
This variable is set automatically per workload and per system according
to [cmdgen metadata](https://github.com/krai/ck-qaic/tree/main/cmdgen).
# CMP40 Huanan boot autoload fix

Working kernel:
6.8.0-139-generic

Root cause:
Forge nvidia.ko depends on kernel module ecc.
At boot nvidia.ko was loaded before ecc and failed with:
- crypto_ecdh_shared_secret
- ecc_is_pubkey_valid_full
- ecc_make_pub_key
- ecc_get_curve
- ecc_gen_privkey

Fix:
forge-cmp-driver.service loads ecc first:

ExecStartPre=/sbin/modprobe ecc

Then:
- forge-cmp/nvidia.ko
- forge-cmp/nvidia-uvm.ko

Verified after clean reboot:
forge-cmp-driver.service = active (exited)
nvidia-smi detects NVIDIA CMP 40HX automatically.

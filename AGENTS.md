# GLM Repo Agent Notes

## Thunderbolt RDMA/JACCL Recovery

Use the live topology as authority. Do not reuse old interface assumptions from
previous runs without rechecking `ifconfig`, `networksetup -listallhardwareports`,
`route`, `arp`, and `ibv_devinfo`.

Current verified two-Mac direct path:

- Local M5: `en1`, MAC `36:13:f1:c0:85:80`, IP `192.168.10.1/30`,
  RDMA device `rdma_en1`, GID `::ffff:192.168.10.1`.
- Peer MAXBOOK: `en1`, MAC `36:a7:c5:18:04:80`, IP `192.168.10.2/30`,
  RDMA device `rdma_en1`, GID `::ffff:192.168.10.2`.
- Peer control SSH is `jackmazac@192.168.100.191`.

When the RDMA/JACCL connection breaks:

1. Keep control traffic on Wi-Fi/Ethernet SSH. Do not depend on the direct
   Thunderbolt IP path for recovery commands.
2. Disable Thunderbolt Bridge on both hosts before JACCL recovery:
   `sudo ifconfig bridge0 down`.
3. Put the direct IPs on the active paired Thunderbolt interfaces. For the
   current cable:
   `sudo ifconfig en1 inet 192.168.10.1 netmask 255.255.255.252 up` locally and
   `sudo ifconfig en1 inet 192.168.10.2 netmask 255.255.255.252 up` on MAXBOOK.
4. Do not add host routes with
   `sudo route -n add -host <peer-ip> -interface <if>`. On macOS this can create
   permanent self-MAC ARP entries and break ping even when IPs and RDMA GIDs
   look correct.
5. If that bad host route exists, delete it and the ARP entry:
   local `sudo route -n delete -host 192.168.10.2`; peer
   `sudo route -n delete -host 192.168.10.1`; then clear ARP if present.
6. Verify ARP before RDMA: local `arp -n 192.168.10.2` must show the peer MAC
   `36:a7:c5:18:04:80`, and peer `arp -n 192.168.10.1` must show the local MAC
   `36:13:f1:c0:85:80`.
7. Verify GIDs: `ibv_devinfo -d rdma_en1 -v` on each host must show
   `PORT_ACTIVE` and `GID[1]` for that host's `192.168.10.x` address.
8. Verify UC pingpong before MLX/JACCL:
   peer `/usr/bin/ibv_uc_pingpong -d rdma_en1 -n 100 -g 1`; local
   `/usr/bin/ibv_uc_pingpong -d rdma_en1 -n 100 -g 1 192.168.10.2`.
9. Verify JACCL without model work using the clean-room preflight. First write
   `/tmp/glm-jaccl-hostfile-en1.json`:

   ```json
   [
     {"ssh": "127.0.0.1", "ips": ["192.168.10.1"], "rdma": [null, "rdma_en1"]},
     {"ssh": "jackmazac@192.168.100.191", "ips": [], "rdma": ["rdma_en1", null]}
   ]
   ```

   Then run:

   ```bash
   GLM_JACCL_HOSTFILE=/tmp/glm-jaccl-hostfile-en1.json \
   GLM_LOCAL_DIRECT_IF=en1 \
   GLM_PEER_DIRECT_IF=en1 \
   GLM_LOCAL_DIRECT_IP=192.168.10.1 \
   GLM_PEER_DIRECT_IP=192.168.10.2 \
   GLM_PEER_SSH=jackmazac@192.168.100.191 \
   GLM_REQUIRED_WIRED_MB=0 \
   GLM_PREFLIGHT_ONLY=1 \
   scripts/run_glm45_air_distributed_cleanroom_cache.sh
   ```

   Leave `GLM_MLX_WIRED_LIMIT_GB` unset by default so MLX/macOS use the normal
   system wired-memory limits. Only set a custom MLX wired limit for an
   explicitly approved diagnostic.

Only after direct ping, RDMA GIDs, UC pingpong, and two-rank JACCL all-sum are
green should a BF16/source teacher-cache run be considered. Model/exporter work
must also preserve the current memory-safety policy; elevated `iogpu.wired_limit_mb`
or custom MLX wired limits are separate, explicitly approved diagnostics.

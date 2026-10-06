from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_cli():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "audit_glm45_air_rdma_topology.py"
    )
    spec = importlib.util.spec_from_file_location("rdma_topology_audit_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_marks_active_rdma_ports_without_direct_ips_not_ready() -> None:
    cli = _load_cli()

    record = cli.build_audit_record(
        local_if="en1",
        peer_if="en1",
        local_ip="192.168.10.1",
        peer_ip="192.168.10.2",
        local_rdma_device="rdma_en1",
        peer_rdma_device="rdma_en1",
        peer_ssh="jackmazac@192.168.100.191",
        local_ifconfig="""en1: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tether 36:13:f1:c0:85:80
\tstatus: active
""",
        peer_ifconfig="""en1: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tether 36:a7:c5:18:04:80
\tstatus: active
""",
        local_route="""route to: 192.168.10.2
destination: default
       mask: default
    gateway: 192.168.100.1
  interface: en0
""",
        peer_route="""route to: 192.168.10.1
destination: default
       mask: default
    gateway: 192.168.100.1
  interface: en0
""",
        local_arp="192.168.10.2 (192.168.10.2) -- no entry\n",
        peer_arp="192.168.10.1 (192.168.10.1) -- no entry\n",
        local_ibv_devinfo="""hca_id: rdma_en1
\t\tstate:\t\t\tPORT_ACTIVE (4)
\t\tGID[  0]:\t\tfe80::3413:f1ff:fec0:8580
""",
        peer_ibv_devinfo="""hca_id: rdma_en1
\t\tstate:\t\t\tPORT_ACTIVE (4)
\t\tGID[  0]:\t\tfe80::34a7:c5ff:fe18:480
""",
        local_hardware_ports="""Hardware Port: Thunderbolt 1
Device: en1
Ethernet Address: 36:13:f1:c0:85:80
""",
        peer_hardware_ports="""Hardware Port: Thunderbolt 1
Device: en1
Ethernet Address: 36:a7:c5:18:04:80
""",
    )

    assert record["record_type"] == "glm45_air_rdma_topology_audit"
    assert record["decision"] == "rdma_link_active_direct_ips_missing"
    assert record["link_layer_ready"] is True
    assert record["direct_path_ready"] is False
    assert record["direct_ip_ready"] is False
    assert record["direct_route_ready"] is False
    assert record["direct_arp_ready"] is False
    assert record["rdma_ipv4_gid_ready"] is False
    assert record["ready_for_uc_pingpong"] is False
    assert record["ready_for_jaccl_preflight"] is False
    assert record["local"]["interface"]["status"] == "active"
    assert record["local"]["interface"]["has_expected_ip"] is False
    assert record["local"]["route"]["interface"] == "en0"
    assert record["local"]["rdma"]["has_ipv4_gid"] is False
    assert record["local"]["hardware_ports"]["Thunderbolt 1"] == {
        "device": "en1",
        "ethernet_address": "36:13:f1:c0:85:80",
    }
    assert record["peer"]["route"]["interface"] == "en0"
    assert record["peer"]["hardware_ports"]["Thunderbolt 1"] == {
        "device": "en1",
        "ethernet_address": "36:a7:c5:18:04:80",
    }
    assert "sudo ifconfig bridge0 down" in record["recovery"]["local_commands"]
    assert "netmask 255.255.255.252" in record["recovery"]["local_commands"]
    assert "route -n add -host" not in record["recovery"]["notes"]


def test_audit_marks_direct_path_ready_when_route_arp_and_gid_match() -> None:
    cli = _load_cli()

    record = cli.build_audit_record(
        local_if="en1",
        peer_if="en1",
        local_ip="192.168.10.1",
        peer_ip="192.168.10.2",
        local_rdma_device="rdma_en1",
        peer_rdma_device="rdma_en1",
        peer_ssh="jackmazac@192.168.100.191",
        local_ifconfig="""en1: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tinet 192.168.10.1 netmask 0xfffffffc broadcast 192.168.10.3
\tether 36:13:f1:c0:85:80
\tstatus: active
""",
        peer_ifconfig="""en1: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tinet 192.168.10.2 netmask 0xfffffffc broadcast 192.168.10.3
\tether 36:a7:c5:18:04:80
\tstatus: active
""",
        local_route="""route to: 192.168.10.2
destination: 192.168.10.0
       mask: 255.255.255.252
  interface: en1
""",
        peer_route="""route to: 192.168.10.1
destination: 192.168.10.0
       mask: 255.255.255.252
  interface: en1
""",
        local_arp="? (192.168.10.2) at 36:a7:c5:18:4:80 on en1 ifscope [ethernet]\n",
        peer_arp="? (192.168.10.1) at 36:13:f1:c0:85:80 on en1 ifscope [ethernet]\n",
        local_ibv_devinfo="""hca_id: rdma_en1
\t\tstate:\t\t\tPORT_ACTIVE (4)
\t\tGID[  1]:\t\t::ffff:192.168.10.1
""",
        peer_ibv_devinfo="""hca_id: rdma_en1
\t\tstate:\t\t\tPORT_ACTIVE (4)
\t\tGID[  1]:\t\t::ffff:192.168.10.2
""",
    )

    assert record["decision"] == "rdma_direct_path_ready"
    assert record["link_layer_ready"] is True
    assert record["direct_path_ready"] is True
    assert record["direct_ip_ready"] is True
    assert record["direct_route_ready"] is True
    assert record["direct_arp_ready"] is True
    assert record["rdma_ipv4_gid_ready"] is True
    assert record["ready_for_uc_pingpong"] is True
    assert record["ready_for_jaccl_preflight"] is True
    assert record["local"]["arp"]["has_expected_mac"] is True
    assert record["peer"]["arp"]["has_expected_mac"] is True

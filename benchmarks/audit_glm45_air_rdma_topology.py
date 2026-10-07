"""Audit the two-Mac GLM-4.5-Air RDMA/JACCL direct path without sudo."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


def _normalize_mac(value: str | None) -> str | None:
    if not value:
        return None
    parts = value.strip().lower().split(":")
    if len(parts) != 6:
        return value.strip().lower()
    return ":".join(part.zfill(2) for part in parts)


def parse_ifconfig(text: str, expected_ip: str) -> dict[str, Any]:
    mac_match = re.search(r"\bether\s+([0-9a-fA-F:]+)", text)
    status_match = re.search(r"\bstatus:\s*(\S+)", text)
    inet_values = re.findall(r"\binet\s+(\d+\.\d+\.\d+\.\d+)", text)
    return {
        "status": status_match.group(1) if status_match else None,
        "mac": _normalize_mac(mac_match.group(1) if mac_match else None),
        "inet": inet_values,
        "has_expected_ip": expected_ip in inet_values,
    }


def parse_route(text: str, expected_if: str) -> dict[str, Any]:
    interface_match = re.search(r"^\s*interface:\s*(\S+)", text, re.MULTILINE)
    gateway_match = re.search(r"^\s*gateway:\s*(\S+)", text, re.MULTILINE)
    interface = interface_match.group(1) if interface_match else None
    return {
        "interface": interface,
        "gateway": gateway_match.group(1) if gateway_match else None,
        "uses_expected_interface": interface == expected_if,
    }


def parse_arp(text: str, expected_mac: str | None) -> dict[str, Any]:
    if "no entry" in text.lower():
        mac = None
    else:
        match = re.search(r"\bat\s+([0-9a-fA-F:]+)\b", text)
        mac = _normalize_mac(match.group(1) if match else None)
    expected = _normalize_mac(expected_mac)
    return {
        "mac": mac,
        "expected_mac": expected,
        "has_expected_mac": bool(mac and expected and mac == expected),
    }


def parse_ibv_devinfo(text: str, expected_ip: str) -> dict[str, Any]:
    return {
        "port_active": "PORT_ACTIVE" in text,
        "has_ipv4_gid": f"::ffff:{expected_ip}" in text,
    }


def parse_hardware_ports(text: str) -> dict[str, dict[str, str | None]]:
    ports: dict[str, dict[str, str | None]] = {}
    current: str | None = None
    for line in text.splitlines():
        if line.startswith("Hardware Port: "):
            current = line.split(": ", 1)[1].strip()
            ports[current] = {"device": None, "ethernet_address": None}
            continue
        if current is None:
            continue
        if line.startswith("Device: "):
            ports[current]["device"] = line.split(": ", 1)[1].strip()
        elif line.startswith("Ethernet Address: "):
            ports[current]["ethernet_address"] = _normalize_mac(
                line.split(": ", 1)[1]
            )
    return ports


def _recovery(
    peer_ssh: str,
    local_if: str,
    peer_if: str,
    local_ip: str,
    peer_ip: str,
) -> dict[str, str]:
    return {
        "local_commands": (
            "sudo ifconfig bridge0 down && "
            f"sudo ifconfig {local_if} inet {local_ip} netmask 255.255.255.252 up"
        ),
        "peer_commands": (
            f"ssh {peer_ssh} 'sudo ifconfig bridge0 down && "
            f"sudo ifconfig {peer_if} inet {peer_ip} netmask 255.255.255.252 up'"
        ),
        "notes": (
            "Do not add host routes. If stale host routes exist, delete them, clear ARP, "
            "then verify route, ARP, RDMA GID[1], UC pingpong, and clean-room JACCL."
        ),
    }


def build_audit_record(
    *,
    local_if: str,
    peer_if: str,
    local_ip: str,
    peer_ip: str,
    local_rdma_device: str,
    peer_rdma_device: str,
    peer_ssh: str,
    local_ifconfig: str,
    peer_ifconfig: str,
    local_route: str,
    peer_route: str,
    local_arp: str,
    peer_arp: str,
    local_ibv_devinfo: str,
    peer_ibv_devinfo: str,
    local_hardware_ports: str = "",
    peer_hardware_ports: str = "",
    command_errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    local_interface = parse_ifconfig(local_ifconfig, local_ip)
    peer_interface = parse_ifconfig(peer_ifconfig, peer_ip)
    local = {
        "interface": local_interface,
        "route": parse_route(local_route, local_if),
        "arp": parse_arp(local_arp, peer_interface["mac"]),
        "rdma": parse_ibv_devinfo(local_ibv_devinfo, local_ip),
        "rdma_device": local_rdma_device,
        "hardware_ports": parse_hardware_ports(local_hardware_ports),
    }
    peer = {
        "interface": peer_interface,
        "route": parse_route(peer_route, peer_if),
        "arp": parse_arp(peer_arp, local_interface["mac"]),
        "rdma": parse_ibv_devinfo(peer_ibv_devinfo, peer_ip),
        "rdma_device": peer_rdma_device,
        "hardware_ports": parse_hardware_ports(peer_hardware_ports),
    }

    link_layer_ready = all(
        (
            local["interface"]["status"] == "active",
            peer["interface"]["status"] == "active",
            local["rdma"]["port_active"],
            peer["rdma"]["port_active"],
        )
    )
    direct_ip_ready = all(
        (
            local["interface"]["has_expected_ip"],
            peer["interface"]["has_expected_ip"],
        )
    )
    direct_route_ready = all(
        (
            local["route"]["uses_expected_interface"],
            peer["route"]["uses_expected_interface"],
        )
    )
    direct_arp_ready = all(
        (
            local["arp"]["has_expected_mac"],
            peer["arp"]["has_expected_mac"],
        )
    )
    rdma_ipv4_gid_ready = all(
        (
            local["rdma"]["has_ipv4_gid"],
            peer["rdma"]["has_ipv4_gid"],
        )
    )
    ready = all(
        (
            link_layer_ready,
            direct_ip_ready,
            direct_route_ready,
            direct_arp_ready,
            rdma_ipv4_gid_ready,
        )
    )
    if ready:
        decision = "rdma_direct_path_ready"
    elif link_layer_ready and (not direct_ip_ready or not rdma_ipv4_gid_ready):
        decision = "rdma_link_active_direct_ips_missing"
    elif local["interface"]["status"] != "active" or peer["interface"]["status"] != "active":
        decision = "rdma_link_down"
    else:
        decision = "rdma_direct_path_not_ready"

    return {
        "record_type": "glm45_air_rdma_topology_audit",
        "decision": decision,
        "link_layer_ready": link_layer_ready,
        "direct_path_ready": ready,
        "direct_ip_ready": direct_ip_ready,
        "direct_route_ready": direct_route_ready,
        "direct_arp_ready": direct_arp_ready,
        "rdma_ipv4_gid_ready": rdma_ipv4_gid_ready,
        "ready_for_uc_pingpong": ready,
        "ready_for_jaccl_preflight": ready,
        "local": local,
        "peer": peer,
        "recovery": _recovery(peer_ssh, local_if, peer_if, local_ip, peer_ip),
        "command_errors": command_errors or [],
    }


def _run(argv: list[str], *, timeout: int) -> tuple[str, dict[str, Any] | None]:
    proc = subprocess.run(
        argv,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    text = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode == 0:
        return text, None
    return text, {"argv": argv, "returncode": proc.returncode, "output": text.strip()}


def collect_live_texts(
    *,
    local_if: str,
    peer_if: str,
    local_ip: str,
    peer_ip: str,
    local_rdma_device: str,
    peer_rdma_device: str,
    peer_ssh: str,
    timeout: int,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    local_ifconfig, error = _run(["ifconfig", local_if], timeout=timeout)
    if error:
        errors.append(error)
    local_hardware_ports, error = _run(["networksetup", "-listallhardwareports"], timeout=timeout)
    if error:
        errors.append(error)
    local_route, error = _run(["route", "-n", "get", peer_ip], timeout=timeout)
    if error:
        errors.append(error)
    local_arp, error = _run(["arp", "-n", peer_ip], timeout=timeout)
    if error:
        errors.append(error)
    local_ibv, error = _run(
        ["/usr/bin/ibv_devinfo", "-d", local_rdma_device, "-v"],
        timeout=timeout,
    )
    if error:
        errors.append(error)

    peer_ifconfig, error = _run(["ssh", peer_ssh, f"ifconfig {peer_if}"], timeout=timeout)
    if error:
        errors.append(error)
    peer_hardware_ports, error = _run(
        ["ssh", peer_ssh, "networksetup -listallhardwareports"],
        timeout=timeout,
    )
    if error:
        errors.append(error)
    peer_route, error = _run(["ssh", peer_ssh, f"route -n get {local_ip}"], timeout=timeout)
    if error:
        errors.append(error)
    peer_arp, error = _run(["ssh", peer_ssh, f"arp -n {local_ip}"], timeout=timeout)
    if error:
        errors.append(error)
    peer_ibv, error = _run(
        ["ssh", peer_ssh, f"/usr/bin/ibv_devinfo -d {peer_rdma_device} -v"],
        timeout=timeout,
    )
    if error:
        errors.append(error)

    return {
        "local_ifconfig": local_ifconfig,
        "peer_ifconfig": peer_ifconfig,
        "local_hardware_ports": local_hardware_ports,
        "peer_hardware_ports": peer_hardware_ports,
        "local_route": local_route,
        "peer_route": peer_route,
        "local_arp": local_arp,
        "peer_arp": peer_arp,
        "local_ibv_devinfo": local_ibv,
        "peer_ibv_devinfo": peer_ibv,
        "command_errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-if", required=True)
    parser.add_argument("--peer-if", required=True)
    parser.add_argument("--local-ip", required=True)
    parser.add_argument("--peer-ip", required=True)
    parser.add_argument("--local-rdma-device", required=True)
    parser.add_argument("--peer-rdma-device", required=True)
    parser.add_argument("--peer-ssh", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit nonzero unless route, ARP, RDMA port, and IPv4 GID checks are ready.",
    )
    args = parser.parse_args()

    texts = collect_live_texts(
        local_if=args.local_if,
        peer_if=args.peer_if,
        local_ip=args.local_ip,
        peer_ip=args.peer_ip,
        local_rdma_device=args.local_rdma_device,
        peer_rdma_device=args.peer_rdma_device,
        peer_ssh=args.peer_ssh,
        timeout=args.timeout,
    )
    record = build_audit_record(
        local_if=args.local_if,
        peer_if=args.peer_if,
        local_ip=args.local_ip,
        peer_ip=args.peer_ip,
        local_rdma_device=args.local_rdma_device,
        peer_rdma_device=args.peer_rdma_device,
        peer_ssh=args.peer_ssh,
        **texts,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output_json)
    if args.require_ready and not bool(record["direct_path_ready"]):
        raise SystemExit(
            "RDMA direct path is not ready; see topology audit evidence at "
            f"{args.output_json}"
        )


if __name__ == "__main__":
    main()

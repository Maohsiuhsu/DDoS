# DDoS

# Paper Title (Under Review)

**A Federated Learning Framework with ADASYN for Detecting ICMP Flood Attacks in Industrial Control Systems**  
📌 Status: **Under review** (currently in the peer-review process)

---

## Overview

This repository provides the implementation for the paper:  
**“A Federated Learning Framework with ADASYN for Detecting ICMP Flood Attacks in Industrial Control Systems”** (under review).

The workflow is:
1. Augment the dataset using **ADASYN** (`adasyn.py`)
2. Start the **federated server** (`server.py`) — you must change the server IP to your own
3. Launch **multiple clients** (`client.py`) — each client must set the IP to the server IP

---

## Quick Start (Run Order)

### Step 1 — Data Augmentation (ADASYN)

Run ADASYN first to expand the dataset:

```bash
python adasyn.py
```bash
### Step 2 — Start the Server (Change IP First)

1. Open server.py and set the server IP to your own machine IP:
2. Run the server:

```bash
python server.py
```bash

### Step 3 — Start Clients (At Least Two, Change IP First)

1. Open client.py and set the IP to the server IP (same as Step 2):
2. Open at least two terminals (or use two machines) and run:

```bash
python client.py
```bash


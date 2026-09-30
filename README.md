\# Attack Diagnosis and Localization in IoT Intrusion Detection Using Formal Modeling and Deep Learning



\## Project Overview



This project extends a formal-modeling-assisted deep learning approach

for IoT intrusion detection by adding attack diagnosis and localization.



The system models IoT device behavior, generates attack scenarios,

captures MQTT traffic, extracts network features, performs intrusion

detection and classification, and uses contextual metadata for attack

diagnosis and localization.



\## Attack Classes



\- Normal

\- MITM

\- Replay

\- Data Falsification

\- Battery Draining



\## Project Pipeline



IoT Simulation

→ MQTT Broker

→ Attack Injection

→ PCAP Capture

→ Feature Extraction

→ Preprocessing

→ IDS

→ Attack Classification

→ Diagnosis

→ Localization

→ Alert



\## Repository Structure



\- `docs/` – project documentation and architecture

\- `models\_fsm/` – TFSM/EFSM definitions and scenario mapping

\- `simulation/` – simulated IoT devices and MQTT traffic

\- `captures/` – packet captures and traffic mapping

\- `features/` – feature extraction and datasets

\- `ids/` – baseline and LSTM models

\- `diagnosis\_loc/` – diagnosis and localization logic

\- `results/` – metrics, confusion matrices and graphs



\## Team



\- Ch. Ram Charan

\- K. Varun

\- A. Varun Satya


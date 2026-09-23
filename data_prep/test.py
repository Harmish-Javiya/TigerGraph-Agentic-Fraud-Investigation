"""
Preprocesses identity.csv into a unique device fingerprint per transaction.

Why this exists: DeviceInfo alone (e.g. "SAMSUNG SM-G892A Build/NRD90M") is
not a unique key — many different customers share the same phone model.
The device_key used everywhere else in this project is the concatenation
of DeviceInfo + id_30 (OS) + id_31 (browser) + id_33 (screen resolution),
which is unique enough to link cards/cases that genuinely share a device.

Run from the folder containing identity.csv:
    python ../data_prep/test.py

Produces two files in the same folder:
    device_profiles.csv   — one row per unique device_key
    txn_device.csv         — TransactionID -> device_key mapping (144k rows)

Both are then loaded into TigerGraph as the DeviceProfile vertex and the
FROM_DEVICE edge (see schema/schema.gsql and the loading jobs in README.md).
graph_client.py also reads txn_device.csv directly at runtime as a fast local
lookup table (get_device_key_for_txn) rather than round-tripping to the graph
for a simple key lookup.
"""

import pandas as pd

df = pd.read_csv('identity.csv')

df['device_key'] = (
    df['DeviceInfo'].fillna('unknown') + '|' +
    df['id_30'].fillna('unknown') + '|' +
    df['id_31'].fillna('unknown') + '|' +
    df['id_33'].fillna('unknown')
)

# Device profiles — one row per unique device
devices = df[
    ['device_key', 'DeviceType', 'DeviceInfo', 'id_30', 'id_31', 'id_33', 'id_23']
].drop_duplicates('device_key')
devices.columns = [
    'device_key', 'device_type', 'device_info',
    'os_name', 'browser_name', 'screen_res', 'proxy_type'
]
devices.to_csv('device_profiles.csv', index=False)

# Transaction -> device mapping
txn_device = df[['TransactionID', 'device_key']]
txn_device.to_csv('txn_device.csv', index=False)

print(f'Unique devices: {len(devices)}')
print(f'Transaction-device mappings: {len(txn_device)}')

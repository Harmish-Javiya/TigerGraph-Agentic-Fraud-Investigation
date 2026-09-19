import pandas as pd

df = pd.read_csv('identity.csv')
df['device_key'] = (
    df['DeviceInfo'].fillna('unknown') + '|' +
    df['id_30'].fillna('unknown') + '|' +
    df['id_31'].fillna('unknown') + '|' +
    df['id_33'].fillna('unknown')
)

# Device profiles - one row per unique device
devices = df[['device_key','DeviceType','DeviceInfo','id_30','id_31','id_33','id_23']].drop_duplicates('device_key')
devices.columns = ['device_key','device_type','device_info','os_name','browser_name','screen_res','proxy_type']
devices.to_csv('device_profiles.csv', index=False)

# Transaction to device mapping
txn_device = df[['TransactionID','device_key']]
txn_device.to_csv('txn_device.csv', index=False)

print(f'Unique devices: {len(devices)}')
print(f'Transaction-device mappings: {len(txn_device)}')

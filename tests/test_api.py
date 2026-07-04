import json
import urllib.request

test_data = {
    'gender': 'Male',
    'senior_citizen': 0,
    'partner': 'No',
    'dependents': 'No',
    'tenure': 12,
    'phone_service': 'Yes',
    'multiple_lines': 'No',
    'internet_service': 'DSL',
    'online_security': 'No',
    'online_backup': 'No',
    'device_protection': 'No',
    'tech_support': 'No',
    'streaming_tv': 'No',
    'streaming_movies': 'No',
    'contract': 'Month-to-month',
    'paperless_billing': 'Yes',
    'payment_method': 'Electronic check',
    'monthly_charges': 70.0,
    'total_charges': 840.0
}

url = 'http://127.0.0.1:5000/api/predict'
req = urllib.request.Request(url, data=json.dumps(test_data).encode('utf-8'), headers={'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode('utf-8')
        print('STATUS', resp.status)
        print('BODY')
        print(body)
except Exception as e:
    print('ERROR', e)
    raise

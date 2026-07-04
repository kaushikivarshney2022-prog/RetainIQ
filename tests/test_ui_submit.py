import urllib.request
import urllib.parse

form_data = {
    'gender': 'Male',
    'senior_citizen': '0',
    'partner': 'No',
    'dependents': 'No',
    'tenure': '12',
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
    'monthly_charges': '70.0',
    'total_charges': '840.0'
}

data = urllib.parse.urlencode(form_data).encode()
req = urllib.request.Request('http://127.0.0.1:5000/predict', data=data, headers={'Content-Type': 'application/x-www-form-urlencoded'})

try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        html = resp.read().decode('utf-8')
        print('STATUS', resp.status)
        with open('tests/result_page.html', 'w', encoding='utf-8') as f:
            f.write(html)
        print('Saved tests/result_page.html')
except Exception as e:
    print('ERROR', e)
    raise

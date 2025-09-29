from cyclic_compression_analysis import analyze_week_data, get_test_name

base_path = '.'
week_folder = '0 WK'
test_name = get_test_name(week_folder)
print(f'Testing {test_name} with updated VS calculation...')
data_by_combination = analyze_week_data(base_path, week_folder)
print(f'Found {len(data_by_combination)} combinations')

found_95_50 = False
for combo_key, combo_data in data_by_combination.items():
    filament = combo_data['filament']
    svf = combo_data['svf']
    vs = combo_data['vs']
    print(f'  - Filament {filament}, SVF {svf}%, VS {vs:.1f} N/mm')
    if filament == 95 and svf == 50:
        found_95_50 = True
        print(f'    ✅ SUCCESS: Found 95-50%!')

if found_95_50:
    print('\n🎉 95-50% data is now included in the analysis!')
else:
    print('\n❌ 95-50% data still not found')
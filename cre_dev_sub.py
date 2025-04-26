import json

# อ่าน dev.json
with open('spider_eval/dev.json', 'r') as f:
    dev_data = json.load(f)

# เลือก 10 คำถามแรก
subset_data = dev_data[:100]

# บันทึกเป็น dev_subset.json
with open('spider_eval/dev_subset_100.json', 'w') as f:
    json.dump(subset_data, f, indent=2)

print("Created dev_subset.json successful!!!")
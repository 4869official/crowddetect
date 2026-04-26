import csv
csv_file_path = 'govinf.csv'
caid="13030200551310196954"
    # 打开CSV文件
with open(csv_file_path, mode='r', encoding='utf-8') as file:
    # 创建CSV阅读器
    reader = csv.DictReader(file)
    fieldnames = reader.fieldnames
    print(fieldnames)

    # 遍历CSV文件的每一行
    for row in reader:
        # 假设CSV文件中有列名为'name'和'phone'

        if row['唯一标识码'] == caid:  # 将'name'替换为你CSV文件中对应的列名
            print(row['唯一标识码'] == caid)
            print(row['唯一标识码'])
            print(caid)
            qwe = row['\ufeff通道名称']
            asd = row['经度']
            zxc = row['纬度']
            rty = row['类别']
            print(qwe)
            print(asd)
            print(zxc)
            print(rty)
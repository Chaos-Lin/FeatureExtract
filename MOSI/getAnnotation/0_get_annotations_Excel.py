import os
import xlwt
import re

# 遍历指定目录下的文件，将文件名按照一定规则写入Excel表格中。

raw_data_path = "E:\\上大-博士\\老人数据集\\原始视频切片\\普通话"
files = os.listdir(raw_data_path)

# 创建新的workbook（其实就是创建新的excel）
workbook = xlwt.Workbook(encoding='ascii')
# 创建新的sheet表
worksheet = workbook.add_sheet("Sheet1")
head = ['video_id', 'clip_id', 'M']
# 循环写入表头
for i in head:
    worksheet.write(0, head.index(i), i)

i = 1
for file in files:
    pattern = re.compile("video_")
    if not pattern.search(file):
        continue
    sub_raw_data_path = raw_data_path + "\\" + file
    sub_files = os.listdir(sub_raw_data_path)
    for sub_file in sub_files:
        worksheet.write(i, 0, file)
        worksheet.write(i, 1, sub_file.split(".")[0])
        i += 1

savePath = 'D:\\SHU\\Dataset\\MESIC\\Annotation.xls'

workbook.save(savePath)

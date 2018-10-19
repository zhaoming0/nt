#!/usr/bin/env python
current_name = None
update_file_list = []
update_expect_value_list = []
mul_flag = False
with open('3.log') as rf:
    for line in rf.readlines():
        if line.find('Start') != -1:
            mul_flag = False
            pos = line.find('example') - 1
            current_name = line[10:pos].replace(' ', '_').lower().replace('same', 'SAME').replace('valid', 'VALID')
            mul_pos = line.find('/')
            if mul_pos != -1:
                mul_num = line[mul_pos+1:line.find('=====Start========')]
                current_name += '_' + mul_num
            if current_name not in update_file_list:
                update_file_list.append(current_name)
            else:
                mul_flag = True
            value_list = []
        elif line.find('End') != -1:
            if mul_flag:
                update_expect_value_list[-1].extend([value_list])
            else:
                update_expect_value_list.append([value_list])
        else:
            value = line.strip('\n')
            if value.isdigit():
                value = int(value)
            else:
                value = float(value)
            value_list.append(value)
index = 0

for file_name in update_file_list:
    print("===============================%d--%s" % (index,file_name))
    source_file = "../test/cts/test/V1_0/" + file_name + ".js"
    dst_file = "../test/cts/supplement_test/" + file_name + "_relu6.js"
    internal_count = 0
    with open(dst_file, 'w') as wf:
        with open(source_file) as srf:
            for sl in srf.readlines():
                if sl.find("it('check result") != -1:
                    update_name = sl[sl.find('for')+4:sl.find("', async")].replace('example', 'relu6 example')
                    sl = "  it('check result for %s', async function() {\n" % update_name
                elif file_name.startswith('conv') and (sl.find("    model.setOperandValue(act, new Int32Array([0]));") != -1 or sl.find("    model.setOperandValue(b7, new Int32Array([0]));") != -1):
                    sl = sl.replace("Int32Array([0", "Int32Array([3")
                elif file_name.startswith('depthwise') and (sl.find("    model.setOperandValue(act, new Int32Array([0]));") != -1 or sl.find("    model.setOperandValue(b8, new Int32Array([0]));") != -1):
                    sl = sl.replace("Int32Array([0", "Int32Array([3")
                elif sl.find('_expect = ') != -1:
                    sl = sl[:sl.find('=')+2] + str(update_expect_value_list[index][internal_count]) + ';\n'
                    internal_count += 1
                wf.write(sl)
    index += 1

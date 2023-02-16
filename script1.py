#!/home/zzq/anaconda3/envs/gprMax/bin/python
"""
本脚本有两个功能，
1、查看.out文件有多少个Dataset
用法    python script.py 0 {example.out}

2、将一个拥有多个接收源生产的.out 文件中的所有Ascan数据合并到一起，使得生成的文件能够绘制Bscan图
用法   
    假如要合成的文件是“example.out”,用下面的命令
    python script.py 1 {example}
    生成的文件是 example_merged.out
"""

import sys
import h5py
import glob
import numpy as np

# 读取HDF5文件中的所有数据集
def traverse_datasets(hdf_file):
    import h5py
 
    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = '{}/{}'.format(prefix, key)
            if isinstance(item, h5py.Dataset): # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group): # test for group (go down)
                yield from h5py_dataset_iterator(item, path)
 
    with h5py.File(hdf_file, 'r') as f:
        for (path, dset) in h5py_dataset_iterator(f):
            print(path, dset)
 
    return None

def eliminate_background(file1, file2, output_file='eilm_bg.out'):
    """对Bscan图的.out文件进行背景对消,然后生成一个背景对消后的.out文件
    参数：
        file1：有要探测目标的文件名
        file2: 无要探测目标的文件名
        output_file: 消除背景后的文件名，默认为"elim_bg.out"
    返回值：
        返回一个消除了背景的.out文件
    """
    fin1 = h5py.File(file1, 'r')
    fin2 = h5py.File(file2, 'r')
    fout = h5py.File(output_file, 'w')

    # 设置输出文件的一些参数
    fout.attrs['Title'] = fin1.attrs['Title']
    fout.attrs['gprMax'] = fin1.attrs['gprMax']
    fout.attrs['Iterations'] = fin1.attrs['Iterations']
    fout.attrs['dt'] = fin1.attrs['dt']
    fout.attrs['nrx'] = 1
    col = fin1['rxs/rx1/Ez'].shape[1]
    for rx in range(1,  2):
        path = '/rxs/rx' + str(rx)
        grp = fout.create_group(path)
        availableoutputs = list(fin1[path].keys())
        for output in availableoutputs:
            grp.create_dataset(output, (fout.attrs['Iterations'], col), dtype=fin1[path + '/' + output].dtype)

    # 进行背景对消
    path = 'rxs/rx1/'
    #for output in list(fin1[path].keys()):
    #    for i in range(fout.attrs['Iterations']):
    #        fout[path + output][i:] = fin1[path + output][i:] - fin2[path + output][i:]

    output = 'Ez'
    for i in range(fout.attrs['Iterations']):
        fout[path + output][i:] = fin1[path + output][i:] - fin2[path + output][i:]

    # 关闭文件
    fin1.close()
    fin2.close()
    fout.close()

def merge_files(basefilename):
    """Merges traces (A-scans) from multiple output files into one new file,
        then optionally removes the series of output files.

    Args:
        basefilename (string): Base name of output file series including path.
        outputs (boolean): Flag to remove individual output files after merge.
    """

    outputfile = basefilename + '_merged.out'
    files = glob.glob(basefilename + '.out')
    outputfiles = [filename for filename in files if '_merged' not in filename]
    modelruns = len(outputfiles)

    # Combined output file
    fout = h5py.File(outputfile, 'w')

    # Add positional data for rxs
    for model in range(1):
        fin = h5py.File(basefilename + '.out', 'r')
        if not fin:
            print("没找到输入文件")
        nrx = fin.attrs['nrx']

        # Write properties for merged file on first iteration
        if model == 0:
            fout.attrs['Title'] = fin.attrs['Title']
            fout.attrs['gprMax'] = fin.attrs['gprMax']
            fout.attrs['Iterations'] = fin.attrs['Iterations']
            fout.attrs['dt'] = fin.attrs['dt']
            fout.attrs['nrx'] = 1
            for rx in range(1,  2):
                path = '/rxs/rx' + str(rx)
                grp = fout.create_group(path)
                availableoutputs = list(fin[path].keys())
                for output in availableoutputs:
                    grp.create_dataset(output, (fout.attrs['Iterations'], nrx), dtype=fin[path + '/' + output].dtype)

        # For all receivers
        for rx in range(1, nrx + 1):
            origin_path = '/rxs/rx1'
            path = '/rxs/rx' + str(rx) 
            availableoutputs = list(fin[path].keys())
            # For all receiver outputs
            for output in availableoutputs:
                fout[origin_path + '/' + output][:, rx-1] = fin[path + '/' + output][:]

        fin.close()

    fout.close()
 
# 传入路径即可
#traverse_datasets(sys.argv[1])

if __name__ == "__main__":
    if sys.argv[1] == "0":
        traverse_datasets(sys.argv[2])
    if sys.argv[1] == "1":
        merge_files(sys.argv[2]) 
    if sys.argv[1] == "2":
        eliminate_background(sys.argv[2], sys.argv[3])


import glob
import sys
import random

import h5py
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from gprMax import gprMax
from tools.outputfiles_merge import get_output_data
from tools.outputfiles_merge import merge_files


def generate_in_file(ifilename: str, ofilename: str, no_need_lines: list) -> bool:
    """
    根据输入文件，生成一个对输入文件注释了一些行的输出文件。
    :param ifilename: 输入的文件名
    :param ofilename: 输出的文件名
    :param no_need_lines: 需要注释的行号
    :return: 是否成功生成文件
    """
    try:
        ofile = open(ofilename + '.in', 'w')
        ifile = open(ifilename + '.in', 'r')
    except OSError:
        print(f"操作文件失败，请查看该文件是否存在或者被占用")
        ofile.close()
        return False
    else:
        for i, line in enumerate(ifile):
            if not (i + 1 in no_need_lines):
                ofile.writelines(line)
        ofile.close()
        ifile.close()
        return True


def generate_in_files(filename: str, type: str = 'all', seed: int = 1000) -> bool:
    """
    生成输入文件
    :param type: all:既有空洞又有目标  obj:只有目标  none:啥都没有
    :return:
    """
    file = open(f'{filename}_{type}.in', 'w')

    random.seed(seed)
    cent_circle_x = random.uniform(0.5, 1)
    cent_circle_x = int(cent_circle_x * 1000) / 1000

    random.seed(seed)
    cent_circle_y = random.uniform(0.45, 0.54)
    cent_circle_y = int(cent_circle_y * 1000) / 1000

    random.seed(seed)
    radius_1 = random.uniform(0.04, 0.06)
    radius_1 = int(radius_1 * 1000) / 1000

    random.seed(seed)
    radius_2 = random.uniform(0.02, 0.04)
    radius_2 = int(radius_2 * 1000) / 1000

    random.seed(seed)
    x_offset = random.uniform(-0.15, 0.15)
    x_offset = int(x_offset * 1000) / 1000

    random.seed(seed)
    y_offset = random.uniform(0.40, 0.55)
    y_offset = int(y_offset * 1000) / 1000

    x1, y1, r1 = cent_circle_x, cent_circle_y, radius_1
    x2, y2, r2 = cent_circle_x + x_offset, cent_circle_y - y_offset, radius_2

    file.write(
        '#domain: 1.500 0.900 0.002\n'
        '#dx_dy_dz: 0.002 0.002 0.002\n'
        '#time_window: 30e-9\n'
        '#material: 6 0 1 0 half_space\n'
        '#material: 81 0.05 1 0 water\n'
        '#material: 4 0.004 1 0 layer2\n'
        '#material: 9 0.005 1 0 layer3\n'
        '#material: 12 0.003 1 0 layer4\n'
        '#material: 3.5 0 1 0 pvc\n'

        '#waveform: ricker 1 800e6 my_ricker\n'
        '#hertzian_dipole: z 0.040 0.800 0 my_ricker\n'
        '#rx: 0.045 0.800 0\n'
        '#src_steps: 0.02 0 0\n'
        '#rx_steps: 0.02 0 0\n'

        '#box: 0 0.8 0 1.5 0.9 0.002 free_space\n'
        '#box: 0 0.7 0 1.5 0.8 0.002 layer2\n'
        '#box: 0 0.60 0 1.5 0.70 0.002 layer3\n'
        '#box: 0 0 0 1.5 0.60 0.002 layer4\n'
    )
    if type == 'all' or type == 'water':
        file.write(
            f'#cylinder: {x1} {y1} 0 {x1} {y1} 0.002 {r1} pvc\n'
            f'#cylinder: {x1} {y1} 0 {x1} {y1} 0.002 {r1 - 0.01} water\n'
        )
    if type == 'all' or type == 'obj':
        offset_x = random.uniform(-0.1, 0.2)
        offset_y = random.uniform(-0.05, 0.05)
        offset_r = random.uniform(-0.005, 0.005)
        file.write(
            f'#cylinder: {x2} {y2} 0 {x2} {y2} 0.002 {r2} pvc\n'
            f'#cylinder: {x2} {y2} 0 {x2} {y2} 0.002 {r2 - 0.01} free_space\n'
            f'#cylinder: {x2 + offset_x} {y2 + offset_y} 0 {x2 + offset_x} {y2 + offset_y} 0.002 {r2 + offset_r} pvc\n'
            f'#cylinder: {x2 + offset_x} {y2 + offset_y} 0 {x2 + offset_x} {y2 + offset_y} 0.002 {r2 - 0.01 + offset_r} free_space\n'
        )
    file.close()
    return True


# generate_in_files("./in/test", 'all')
# generate_in_files("./in/test", 'obj')
# generate_in_files("./in/test", 'none')


def real_aperture_merge_files(basefilename):
    """将用实孔径方法扫描出的文件合成为能画出Bscan图像的格式"""
    outputfile = basefilename + '_merged.out'
    files = glob.glob(basefilename + '.out')

    # Combined output file
    fout = h5py.File(outputfile, 'w')

    # Add positional data for rxs
    for model in range(1):
        print(basefilename)
        fin = h5py.File(basefilename + '.out', 'r')
        # sys.exit(0)
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
            for rx in range(1, 2):
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
                fout[origin_path + '/' + output][:, rx - 1] = fin[path + '/' + output][:]

        fin.close()

    fout.close()


def eliminate_background(file1, file2, output_file='eilm_bg.out'):
    """对Bscan图的.out文件进行背景对消,然后生成一个背景对消后的.out文件
    参数：
        file1：有要探测目标的文件名
        file2: 无要探测目标的文件名
        output_file: 消除背景后的文件名，默认为"elim_bg.out"
    返回值：
        返回一个消除了背景的.out文件
    """
    fin1 = h5py.File(file1 + '.out', 'r')
    fin2 = h5py.File(file2 + '.out', 'r')
    fout = h5py.File(output_file, 'w')

    # 设置输出文件的一些参数
    fout.attrs['Title'] = fin1.attrs['Title']
    fout.attrs['gprMax'] = fin1.attrs['gprMax']
    fout.attrs['Iterations'] = fin1.attrs['Iterations']
    fout.attrs['dt'] = fin1.attrs['dt']
    fout.attrs['nrx'] = 1
    col = fin1['rxs/rx1/Ez'].shape[1]
    for rx in range(1, 2):
        path = '/rxs/rx' + str(rx)
        grp = fout.create_group(path)
        availableoutputs = list(fin1[path].keys())
        for output in availableoutputs:
            grp.create_dataset(output, (fout.attrs['Iterations'], col), dtype=fin1[path + '/' + output].dtype)

    # 进行背景对消
    path = 'rxs/rx1/'

    output = 'Ez'
    for i in range(fout.attrs['Iterations']):
        fout[path + output][i:] = fin1[path + output][i:] - fin2[path + output][i:]

    # 关闭文件
    fin1.close()
    fin2.close()
    fout.close()


def run(basefilename: str, nums: int) -> bool:
    for i in tqdm(range(nums)):
        # generate_in_file(basefilename, basefilename + f'_{i}', [])
        gprMax.api(basefilename + f'.in', n=57)
        merge_files(basefilename, removefiles=True)
        generate_in_file(basefilename, basefilename + f'_noobj', [20, 21, 23])
        gprMax.api(basefilename + '_noobj.in', n=57)
        # sys.exit(0)
        merge_files(basefilename + '_noobj', removefiles=True)
        eliminate_background(basefilename + '_merged', basefilename + '_noobj_merged', f'eilm_bg{i}.out')
        rxnumber = 1
        rxcomponent = 'Ez'
        outputdata, dt = get_output_data(f'eilm_bg{i}.out', rxnumber, rxcomponent)
        plt.imshow(outputdata, extent=[0, outputdata.shape[1], outputdata.shape[0], 0], interpolation='nearest',
                   aspect='auto', cmap='gray',
                   vmin=-np.amax(np.abs(outputdata)), vmax=np.amax(np.abs(outputdata)))
        plt.savefig(f"./img/{i + 8}.png", dpi=300)
        plt.show()

def remove_direct_wave(data: np.array) -> np.array:
    for i, row in enumerate(data):
        data[i, :] = row - np.mean(row)
    return data

def save_bscan_img(filename: str, pos: str, dir_path: str = './img') -> bool:
    from skimage import transform
    rxnumber = 1
    rxcompoent = 'Ez'
    outputdata, dt = get_output_data(f'{filename}.out', rxnumber, rxcompoent)
    # outputdata = remove_direct_wave(outputdata)
    outputdata = transform.resize(outputdata, (256, 256))
    # outputdata[:45, :] = 0
    plt.imshow(outputdata, cmap=matplotlib.colormaps['gray'])
    plt.imsave(f'{dir_path}/{pos}.jpg', outputdata, cmap=matplotlib.colormaps['gray'])
    plt.show()
    # plt.axis('off')
    # plt.xticks([])
    # plt.savefig(f'{dir_path}/{filename.split("/")[2].split("_")[0]}{filename.split(".")[0].split("bg")[-1]}{pos}.jpg',
    #             bbox_inches='tight', pad_inches=0)
    # plt.show()

#save_bscan_img("./out/water_eilm_bg0", 'water_0', './imgs')

def start(basename: str, n: int, begin: int, end: int) -> bool:
    for i in range(begin, end):
        seed = random.randint(0, 10000)
        generate_in_files(f'./in/{basename}{i}', 'all', seed)
        generate_in_files(f'./in/{basename}{i}', 'water', seed)
        generate_in_files(f'./in/{basename}{i}', 'obj', seed)
        generate_in_files(f'./in/{basename}{i}', 'none', seed)
        gprMax.api(f'./in/{basename}{i}_all.in', n, gpu=[0])
        merge_files(f'./in/{basename}{i}_all', removefiles=True)
        gprMax.api(f'./in/{basename}{i}_water.in', n, gpu=[0])
        merge_files(f'./in/{basename}{i}_water', removefiles=True)
        gprMax.api(f'./in/{basename}{i}_obj.in', n, gpu=[0])
        merge_files(f'./in/{basename}{i}_obj', removefiles=True)
        gprMax.api(f'./in/{basename}{i}_none.in', n, gpu=[0])
        merge_files(f'./in/{basename}{i}_none', removefiles=True)
        eliminate_background(f'./in/{basename}{i}_all_merged', f'./in/{basename}{i}_none_merged',
                             f'./out/all_eilm_bg{i}.out')
        eliminate_background(f'./in/{basename}{i}_obj_merged', f'./in/{basename}{i}_none_merged',
                             f'./out/obj_eilm_bg{i}.out')
        eliminate_background(f'./in/{basename}{i}_water_merged', f'./in/{basename}{i}_none_merged',
                             f'./out/water_eilm_bg{i}.out')

        save_bscan_img(f'./out/all_eilm_bg{i}', f'x_all_{i}', './imgs')
        save_bscan_img(f'./out/water_eilm_bg{i}', f'water_{i}', './imgs')
        save_bscan_img(f'./out/obj_eilm_bg{i}', f'obj_{i}', './imgs')
        # rxnumber = 1
        # rxcomponent = 'Ez'
        # outputdata1, dt = get_output_data(f'./out/all_eilm_bg{i}.out', rxnumber, rxcomponent)
        # outputdata2, dt = get_output_data(f'./out/obj_eilm_bg{i}.out', rxnumber, rxcomponent)
        # plt.imshow(outputdata1, extent=[0, outputdata1.shape[1], outputdata1.shape[0], 0], interpolation='none',
        #           aspect='auto', cmap='gray',
        #           vmin=-np.amax(np.abs(outputdata1)), vmax=np.amax(np.abs(outputdata1)))
        # plt.savefig(f"./img/{basename}{i}_x.png", dpi=300)
        # plt.show()
        # plt.imshow(outputdata2, extent=[0, outputdata2.shape[1], outputdata2.shape[0], 0], interpolation='none',
        #           aspect='auto', cmap='gray',
        #           vmin=-np.amax(np.abs(outputdata2)), vmax=np.amax(np.abs(outputdata2)))
        # plt.savefig(f"./img/{basename}{i}_y.png", dpi=300)
        # plt.show()


# run('test', 1)
"""
def save_bscan_img(filename: str, dir_path: str = './img') -> bool:
    from PIL import Image
    from skimage import transform
    rxnumber = 1
    rxcompoent = 'Ez'
    outputdata, dt = get_output_data(f'{filename}.out', rxnumber, rxcompoent)
    # np.savetxt('out2.csv', outputdata, delimiter=',')
    print(outputdata.shape)
    outputdata[:640, :] = 0
    # outputdata[(-0.5 < outputdata) & (outputdata < 0.5)] = 0
    outputdata = transform.resize(outputdata, (256, 256))
    plt.imshow(outputdata, cmap=matplotlib.colormaps['gray'])
    plt.axis('off')
    plt.xticks([])
    plt.savefig(f'{dir_path}/{filename.split("/")[-1]}.jpg', bbox_inches='tight', pad_inches=0)
    plt.show()
    # img.show()
"""


def preprocess(filename: str):
    data = np.loadtxt(filename, delimiter=',')
    for row in data:
        if row.sum():
            row = row / row.sum()
    # data[(data > -20) & (data < 20)] = 0
    data[:200, :] = 0
    plt.imshow(data, cmap=matplotlib.colormaps['gray'])
    plt.axis('off')
    plt.xticks([])
    plt.savefig('9.jpg', bbox_inches='tight', pad_inches=0)
    plt.show()


def minus_z(x):
    m, n = x.shape
    q = np.zeros((m, n))
    for j in range(n):
        q[j, :] = x[j, :] - np.mean(x[j, :])
    return q


def load_mat(filename: str, pos: int):
    import scipy.io as scio
    data = scio.loadmat(filename)['xa']
    print(data.shape)
    from skimage import transform
    data = transform.resize(data, (256, 256))
    # data = minus_z(data)
    plt.imshow(data, cmap=matplotlib.colormaps['gray'])
    plt.axis('off')
    plt.xticks([])
    plt.savefig(f'real_test/{pos}.jpg', bbox_inches='tight', pad_inches=0)

    plt.show()


def add_realdata(file1, file2):
    import scipy.io as scio
    from skimage import transform
    data1 = scio.loadmat(file1)['xa']
    data2 = scio.loadmat(file2)['xa']
    data1 = transform.resize(data1, (256, 256))
    data2 = transform.resize(data2, (256, 256))
    data1 = minus_z(data1)
    data2 = minus_z(data2)
    data = data1 + data2
    # data = minus_z(data)
    plt.imshow(data, cmap=matplotlib.colormaps['gray'])
    plt.axis('off')
    plt.xticks([])
    plt.savefig(f'aaa.jpg', bbox_inches='tight', pad_inches=0)

    plt.show()


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def remove_mult(data):
    row, col = data.shape
    for i in range(col):
        flag = 0
        for j in range(row):
            if data[j, i] > 0.65:
                flag = 1
            if flag == 1 and data[j, i] > 0.4 and data[j, i] < 0.6:
                data[j, i] = 0.54
    return data


def process_realdata(file1):
    import scipy.io as scio
    from skimage import transform
    data = scio.loadmat(file1)['xa']
    data = transform.resize(data, (256, 256))
    data = minus_z(data)
    data = normalization(data)
    data = remove_mult(data)
    plt.imshow(data, cmap=matplotlib.colormaps['gray'])
    plt.axis('off')
    plt.xticks([])
    plt.show()

    # if __name__ == "__main__":
    """
    begin = int(sys.argv[1])
    end = int(sys.argv[2])
    start('test', begin, end)
    """
    # save_bscan_img(f'./out/all_eilm_bg201')
    # save_bscan_img(f'F:/zhiqin/simulation_test/test2_merged', f'./test_data')
    # preprocess('out1.csv')
    # for i in range(16, 31):
    #     load_mat(f'F:\zhiqin\gpr_tx01\TX01__0{i}.mat', i-15)
    # add_realdata(f'F:\zhiqin\gpr_tx01\TX01__016.mat', f'F:\zhiqin\gpr_tx01\TX01__018.mat')
    # process_realdata(f'F:\zhiqin\gpr_tx01\TX01__016.mat')


start("third", 72, 11, 50)
#save_bscan_img('test1_merged', 't13')
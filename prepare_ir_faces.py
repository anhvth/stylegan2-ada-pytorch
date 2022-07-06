from avcv.all import *

SYM = './datasets/ir_faces'
os.system(f'rm -r {SYM}')

mmcv.mkdir_or_exist(SYM)
np.random.seed(0)


symlink = []
metas = [
    '/data/RLDD/all_video_symlink/meta.csv',
    '/data/DMS_Drowsiness/all_video_symlink/meta.csv',
]
for path in metas:
    df = pd.read_csv(path)
    symlink.extend(df.video_symlink.values.tolist())
croped_folders = []
for p in symlink:
    p = osp.splitext(p)[0]
    possible_croped_folder = osp.join(p, 'croped_faces')
    if osp.exists(possible_croped_folder):
        croped_folders.append(possible_croped_folder)

# import ipdb; ipdb.set_trace()
def f(inp):
    i, folder = inp
    paths = glob(osp.join(folder, '*.jpg'))
    
    n = min(500, len(paths))
    rpaths = np.random.choice(paths, n, replace=False)
    for j, path in enumerate(rpaths):

        new_path = osp.join(SYM, f'{i:04d}/{osp.basename(path)}')
        new_path = osp.abspath(new_path)
        img = mmcv.imread(path, 0)
        img = mmcv.imresize(img, (256, 256))
        mmcv.imwrite(img, new_path)
        # os.system(f'ln -s {osp.abspath(path)} {new_path}')
        

from multiprocessing import Pool
from tqdm import tqdm

with Pool(10) as p:
    it = p.imap(f, enumerate(croped_folders))
    total = len(croped_folders)
    r = list(tqdm(it, total=total))
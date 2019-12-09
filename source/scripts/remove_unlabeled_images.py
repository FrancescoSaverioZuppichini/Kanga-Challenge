from pathlib import Path
from argparse import Argument

parser = argparse.ArgumentParser()
parser.add_argument('--root',
                    type=str,
                    default='train',
                    help='dir that contains the images.')
args = parser.parse_args()

root = Path(args.root)

images = root.glob('*.jpg')
labels = root.glob('*.txt')

images_id = [x.name.split('.')[0] for x in images]
labels_id = [x.name.split('.')[0] for x in labels]

missing = list(set(images_id) - set(labels_id))

print(f'{len(missing)} iamges without labels.')

for miss in missing:
    p = root / f'{miss}.jpg'
    p.unlink()
    print(f'Removed {str(p)}')
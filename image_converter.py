# convert all files into webp  from cli arg

import argparse
import os
import tqdm
import pathlib
import cv2
image_types = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp', 'avif', 'heif', 'heic']
def main():
    '''Run if main module'''
    parser = argparse.ArgumentParser(description='Convert all files into webp')
    # dir
    parser.add_argument('dir', help='Directory to convert')
    # format
    parser.add_argument('--into', default='jpg', help='Format to convert to')

    args = parser.parse_args()
    print(f"Converting images in {args.dir} into {args.into}")
    # prefix `into`with period 
    if not args.into.startswith('.'):
        args.into = '.' + args.into
    # get all files in dir
    files = [p for p in pathlib.Path(args.dir).rglob(f"*") if p.is_file()]
    # if file is an image and not webp, convert to webp
    for file in tqdm.tqdm(files):
        if file.suffix == args.into:
            continue

        try:
            if file.suffix[1:] in image_types:
                img = cv2.imread(str(file))
                cv2.imwrite(str(file.with_suffix(args.into)), img)
                os.remove(file)
                print(f"Converted {file} to {args.into}")
        except Exception as e:
            print(f"Error converting {file}: {e}")
            continue

if __name__ == '__main__':
    main()





def main():
    '''Run if main module'''
    # get all files in dir
    from argparse import ArgumentParser
    from pathlib import Path
    # check filenames from 001 to 100 for what i missed
    start = 1
    end = 101

    parser = ArgumentParser(description='Check for missed files')
    # dir
    parser.add_argument('dir', help='Directory to check')

    args = parser.parse_args()
    print(f"Checking files in {args.dir}")
    # get all files in dir
    files = [p for p in Path(args.dir).rglob(f"*") if p.is_file()]
    # check for missed files

    for i in range(start, end):
        filename = f"{i:03d}"
        if not any(p.stem == filename for p in files):
            print(f"Missed {filename}")
        # if file with that stem is not in files, print it
    

if __name__ == '__main__':
    main()

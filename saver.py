# 
import uuid
import cv2
import numpy as np

BLACK = [0, 0, 0]
WHITE = [255, 255, 255]
def from_binary_matrix_to_bitmap(binary_matrix:list[list[int]]) -> np.ndarray:
  # Transpose the binary matrix to correct the orientation
    transposed_matrix = np.transpose(binary_matrix)
    
    # Convert transposed binary matrix to bitmap
    bitmap = np.zeros((len(transposed_matrix), len(transposed_matrix[0]), 3), dtype=np.uint8)
    for i in range(len(transposed_matrix)):
        for j in range(len(transposed_matrix[0])):
            if transposed_matrix[i][j] == 1:
                bitmap[i][j] = BLACK
            else:
                bitmap[i][j] = WHITE
    return bitmap

def xyn_to_bitmap(xyn:list[list[float]]) -> np.ndarray:
    GRID = 48
    grid = [[0 for _ in range(GRID)] for _ in range(GRID)]
    for x, y, _ in xyn:
        i = int(x * GRID)
        j = int(y * GRID)
        grid[i][j] = 1
    return from_binary_matrix_to_bitmap(grid)

dir = "./datasets/spells/"
def save(bitmap:list[list[int]], classification_idx:int):
    # save into a bitmap file
    filename = f"{dir}{uuid.uuid4()}.{classification_idx}.bmp"
    cv2.imwrite(filename, from_binary_matrix_to_bitmap(bitmap))
    print(f"Saved as {filename}")

def main():
    '''Run if main module'''
    # count each spell class
    import pathlib
    spell_count = [0, 0, 0]

    # scan `dir` for existing files
    p = pathlib.Path(dir)
    for file in p.iterdir():
        if file.is_file():
            spell_count[int(file.name.split('.')[-2])] += 1
            # spell_count[int(file.name.split('.')[-1][0])] += 1

    print(f"Spell count: {spell_count}")

if __name__ == '__main__':
    main()

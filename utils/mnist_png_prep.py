import os
import argparse

parser = argparse.ArgumentParser(
    prog='Code to prepare image and label files for mnist_png')
parser.add_argument('-P', '--parent_folder', required=True,
                    help='Path to where mnist_png has been unpacked.')
parser.add_argument('-O', '--output_folder', required=False,
                    default=os.getcwd(),
                    help='Path where the image and label files will be stored.')

mnist_map = {
    '0': 'Zero',
    '1': 'One',
    '2': 'Two',
    '3': 'Three',
    '4': 'Four',
    '5': 'Five',
    '6': 'Six',
    '7': 'Seven',
    '8': 'Eight',
    '9': 'Nine'
}


def prepare_files(parent_folder, output_folder, mode):
    if mode not in ('training', 'testing'):
        raise ValueError(
            'The argument mode must be one of training or testing.')

    fid_img = open(os.path.join(output_folder, 'img_{}.txt'.format(mode)), 'w')
    fid_label = open(os.path.join(output_folder, 'lbl_{}.txt'.format(mode)),
                     'w')

    parent_folder = os.path.join(parent_folder, mode)
    for d in os.listdir(parent_folder):
        search_path = os.path.join(parent_folder, d)
        for f in os.listdir(search_path):
            fid_img.write('{}\n'.format(os.path.join(search_path, f)))
            fid_label.write('{}\n'.format(str(d)))

    fid_img.close()
    fid_label.close()
    return None


if __name__ == "__main__":
    args = parser.parse_args()
    parent_folder = args.parent_folder
    output_folder = args.output_folder
    if not os.path.exists(parent_folder):
        raise ValueError('The folder {} does not exist.'.format(parent_folder))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    prepare_files(parent_folder, output_folder, 'training')
    prepare_files(parent_folder, output_folder, 'testing')
    fid_map = open(os.path.join(output_folder, 'mnist_map.txt'), 'w')
    for key, value in mnist_map.items():
        fid_map.write('{}\t{}\n'.format(key, value))

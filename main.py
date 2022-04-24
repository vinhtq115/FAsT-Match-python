from argparse import ArgumentParser

import cv2
import time

from fast_match import FAsTMatch


def main(args):
    # Read image
    input_image_file = args.image_file
    template_image_file = args.template_file
    input_image = cv2.imread(input_image_file)
    output_image = input_image.copy()
    template_image = cv2.imread(template_image_file)

    fast_match = FAsTMatch(0.15, 0.85, False, 0.5, 2.0)
    start = time.time()
    corners, distance = fast_match.apply(input_image, template_image)
    end = time.time()
    print(f'Duration: {end - start:.3f}s')

    cv2.polylines(output_image, [corners.astype(int)], True, (0, 0, 255), 2)

    if args.output is not None:
        cv2.imwrite(args.output, output_image)

    cv2.imshow('Input', input_image)
    cv2.imshow('Output', output_image)
    cv2.imshow('Template', template_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = ArgumentParser(description='FAsT-Match')
    parser.add_argument('image_file', help='Input image file', type=str)
    parser.add_argument('template_file', help='Template image file', type=str)

    parser.add_argument('-o', '--output', help='Path to output image', type=str, required=False)

    main(parser.parse_args())

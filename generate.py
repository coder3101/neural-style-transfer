"""
    Copyright 2018 Ashar <ashar786khan@gmail.com>
 
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
 
        http://www.apache.org/licenses/LICENSE-2.0
 
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
import os
import warnings
from argparse import ArgumentParser, ArgumentTypeError

from transfer import NeuralStyleTransfer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter('ignore', category=FutureWarning)

def get_build_model_from_args(arg):
    return NeuralStyleTransfer(content_pth=arg['content_image'],
                               style_pth=arg['style_image'],
                               gen_file_name=arg['gen_file_name'],
                               epochs=arg['epochs'],
                               initialize_randomly=arg['random_init'],
                               learning_rate=arg['learning_rate'],
                               content_factor=arg['content_factor'],
                               style_factor=arg['style_factor'],
                               optimizer=arg['optimizer'],
                               loss_after=arg['loss_after'])


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def main():
    parser = ArgumentParser()
    parser.add_argument('--content_image',
                        type=str,
                        required=True,
                        help='The Path of the image file in any image format. The resultant image will resemble this.')
    parser.add_argument('--style_image',
                        type=str,
                        required=True,
                        help='The Path of the style image to apply to Content image.')
    parser.add_argument('--gen_file_name',
                        type=str,
                        default='./styled_image.jpg',
                        help='Name of Generated File including extension')
    parser.add_argument('--epochs',
                        type=int,
                        default=5000,
                        help='Number of epoch to train')
    parser.add_argument("--random_init",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default='N',
                        help="Generates Image from Scratch if flagged")
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.5,
                        help='Learning rate of model')
    parser.add_argument("--content_factor",
                        type=float,
                        default=0.001,
                        help="Amount of Content to Capture or Optimize upon")
    parser.add_argument("--style_factor",
                        type=float,
                        default=1.0,
                        help="Amount of Style to Capture or Optimize upon")
    parser.add_argument('--optimizer',
                        type=str,
                        default='adam',
                        help='Optimizer : adam or rms')
    parser.add_argument('--loss_after',
                        type=int,
                        default=100,
                        help='Prints loss after this much epochs')
    args = parser.parse_args()
    darg = vars(args)
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('--> 1. Building Computation Graph')
    model = get_build_model_from_args(darg)
    print('--> 2. Training for : ' + str(darg['epochs']) + ' epochs')
    print('--> 3. You will be shown Loss of the image after every : ' +
          str(darg['loss_after']) + ' epochs')
    print('--> 4. Using Randomly Initialized Image : ', darg['random_init'])
    print('--> 5. Style Factor is : ' + str(darg['style_factor']))
    print('--> 6. Content Factor is :' + str(darg['content_factor']))
    print('--> 7. Learning rate will be : ' + str(darg['learning_rate']))
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')


    model.generate()


if __name__ == "__main__":
    main()

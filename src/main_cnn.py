import argparse
from glob import glob

import tensorflow as tf

from model_cnn import denoiser
from utils_cnn import *

#command line format
# python3 main_cnn.py --<option_name> <value> ....

#================= BEGIN: PARSE ARGUMENTS =================#

parser = argparse.ArgumentParser(description='')

#---VS: Following arguments irrspective of training or testing and irrespective of image format----
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--sigma', dest='sigma', type=int, default=25, help='noise level')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here or loaded from here')
parser.add_argument('--img_format', dest='img_format', default='png', help='Format for validation / test images. Choices are png or bin. \
                                                                            For training we use patches within the .npy file specified by --patches_pyfile')

#---VS: Following arguments are for phase==training and are irrespective of img_format---
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='# of epoch for training')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.0005, help='initial learning rate for adam')
parser.add_argument('--patches_pyfile', dest='patches_pyfile', default='img_clean_pats', help='.npy file in --data_dir listing all trainining patches (4-D array), range 0 to 255 \
                                                                                               Generate this file using generate_patches.py .')
parser.add_argument('--data_dir', dest='data_dir', default='./data',help='File with .npy extension containing all training patches is in this directory. \
                                                                          If img_format is png, then all validation / test datasets must be in this directory')

#----VS: Following arguments are for phase==training and img_format==png---
parser.add_argument('--eval_set', dest='eval_set', default='Set12', help='png dataset for validation in training')

#----VS: Following arguments are for img_format==png and are irresepective of whether phase == training or test ---
parser.add_argument('--sample_dir', dest='sample_dir', default='./samples', help='validation and test samples are saved here under sub-directories val and test respectively')

#----VS: Following arguments are for img_format==bin and are irresepective of whether phase == training or test ---
#----VS: If phase==train, the below arguments specify the files for validation images, else if phase==test then the below arguments specify files for test images---
parser.add_argument('--img_filename_base_in',  dest='img_filename_base_in', default='HighClutter1', help='Files are represented by <img_filename_base>_slice<SliceIndex>.2Dimgdata')
parser.add_argument('--img_filename_base_out', dest='img_filename_base_out', default='HighClutter1_out', help='Files are represented by <img_filename_base>_slice<SliceIndex>.2Dimgdata')
parser.add_argument('--Nz', dest='Nz', type=int, default='1', help='Number of slices')
parser.add_argument('--Ny', dest='Ny', type=int, default='512', help='Number of slices')
parser.add_argument('--Nx', dest='Nx', type=int, default='512', help='Number of slices')
parser.add_argument('--SliceNumDigits', dest='SliceNumDigits', type=int, default='4', help='Number of digits to represent slice index')
parser.add_argument('--FirstSliceNumber', dest='FirstSliceNumber', type=int, default='1', help='First slice index')


#----VS: Following arguments are for phase==testing and img_format==png---
parser.add_argument('--test_set', dest='test_set', default='BSD68', help='png dataset for testing')

#----VS: Following arguments are for phase==testing and img_format==bin ---
parser.add_argument('--use_bounds', dest='use_bounds', type=int, default=0, help='0 - Compute pixel bounds on-the-fly, 1 - Use bounds specified by --lower_bound and --upper_bound')
parser.add_argument('--lower_bound', dest='lower_bound', type=float,  default=0, help='Lower range of pixel value')
parser.add_argument('--upper_bound', dest='upper_bound', type=float,  default=255, help='Upper range of pixel value')


#----VS: Following arguments are for phase==testing and irrespective of img_format ---
parser.add_argument('--is_add_noise', dest='is_add_noise', type=int, default=0, help='Treat test images as already noisy (add_noise=0) or clean (add_noise=1)')
parser.add_argument('--ckpt_state', dest='ckpt_state', default='latest', help='Use latest for most recent epoch or specific for a particular epoch')
parser.add_argument('--ckpt_epoch_num', dest='ckpt_epoch_num', type=int, default=0, help='If ckpt_state is set to specific, then this specifies epoch number for checkpoint')

args = parser.parse_args()

#================= END: PARSE ARGUMENTS =================#


def denoiser_train(denoiser, lr):
    #creates object data=train_data(filepath)
    with load_data(filepath=(args.data_dir + '/' + args.patches_pyfile + '.npy')) as data: 
        # After 'with' statement __enter__ within class train() is executed ...
        
        # This loads the data, i.e. an array of patches from a single numpy file specified by "filepath".
        # if there is a small memory, please comment this line and uncomment the line99 in model.py
        data = data.astype(np.float32) / 255.0  # normalize the data to 0-1

        # Files containing validation images
        eval_files=[]
        if(args.img_format=='png'):
            eval_files = glob( (('%s/{}/*.png') % args.data_dir).format(args.eval_set))
        else:
            for i in range(args.Nz):
                fname = '%s_slice%0*d.2Dimgdata' % (args.img_filename_base_in, args.SliceNumDigits, args.FirstSliceNumber+i)
                eval_files.append(fname)
        
        # Read validation images and file-names (not fully complete) for denoiser output
        # Note eval_data is a list of images, each of size (1,Ny,Nx,1)
        out_files = []
        if(args.img_format=='png'):
            eval_data = load_images_png(eval_files)
            vl, vh = None, None 
            for idx in range(len(eval_data)):
                fname = os.path.join((args.sample_dir + '/val'), 'test_img%0*d' % (NumDigitsImg, idx+1))
                out_files.append(fname)
        else:
            eval_data, vl, vh = load_images_bin(eval_files, args.Nz, args.Ny, args.Nx)
            for idx in range(len(eval_data)):
                fname = '%s_slice%0*d' % (args.img_filename_base_out, args.SliceNumDigits, args.FirstSliceNumber+i)
                out_files.append(fname)

        # Train        
        denoiser.train(data, eval_data, batch_size=args.batch_size, ckpt_dir=args.ckpt_dir, epoch=args.epoch, lr=lr,
                       sample_files=out_files, img_format=args.img_format, lower_bound=vl, upper_bound=vh) #out_dir can be 
        
        #after this __exit__ in class train() is invoked, deletes data that was loaded for training 

def denoiser_test(denoiser):
    test_files=[]
    # Test files
    if(args.img_format=='png'):
        test_files = glob( (('%s/{}/*.png') % args.data_dir).format(args.test_set) ) 
    else:
        for i in range(args.Nz):
            fname = '%s_slice%0*d.2Dimgdata' % (args.img_filename_base_in, args.SliceNumDigits, args.FirstSliceNumber+i)
            test_files.append(fname) 

    # Read test images and file-names (not fully complete) for denoiser output
    # Note test_data is a list of images, each of size (1,Ny,Nx,1)        
    out_files = []
    if(args.img_format=='png'):
        test_data = load_images_png(test_files)
        lower_bound, upper_bound = None, None 
        for idx in range(len(test_data)):
            fname = os.path.join((args.sample_dir + '/test'), 'test_img%0*d' % (NumDigitsImg, idx+1))
            out_files.append(fname)
    else:
        test_data, vl, vh = load_images_bin(test_files, args.Nz, args.Ny, args.Nx)
        for idx in range(len(test_data)):
            fname = '%s_slice%0*d' % (args.img_filename_base_out, args.SliceNumDigits, args.FirstSliceNumber+idx)
            out_files.append(fname)

        # Lower and upper bounds on pixel values in Test images       
        if(args.use_bounds):
            lower_bound = args.lower_bound * np.ones([len(test_data)]) 
            upper_bound = args.upper_bound * np.ones([len(test_data)])
        else:
            lower_bound, upper_bound = vl, vh        
     
    denoiser.test(test_data=test_data, ckpt_dir=args.ckpt_dir, save_files=out_files, img_format=args.img_format, ckpt_state=args.ckpt_state, 
                  is_add_noise=bool(args.is_add_noise), ckpt_epoch_num=args.ckpt_epoch_num, lower_bound=lower_bound, upper_bound=upper_bound)

def main(_):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    
    if (args.img_format=='png'):
        if not os.path.exists(args.sample_dir):
            os.makedirs(args.sample_dir)
        #sub-directories
        if not os.path.exists(args.sample_dir+'/val'):
            os.makedirs(args.sample_dir+'/val')
        if not os.path.exists(args.sample_dir+'/test'):
            os.makedirs(args.sample_dir+'/test')


    # decrease learning rate with #epochs (this is a step function, smoother fn may be better)    
    lr = args.lr * np.ones([args.epoch])
    lr[30:] = lr[0] / 10.0

    if(args.phase=='training'):
        print("sigma=%.2f and learing-rate=%.5f" % (args.sigma, args.lr))

    if args.use_gpu:
        # added to control the gpu memory
        print("GPU\n")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            #build the computational graph first, prior to training or testing 
            #In latter case you can load the parameters from latest checkpoint (.ckpt file). This requires knowledge pof comp graph 
            model = denoiser(sess, sigma=args.sigma) 
            if args.phase == 'train':
                denoiser_train(model, lr=lr)
            elif args.phase == 'test':
                denoiser_test(model)
            else:
                print('[!]Unknown phase')
                exit(0)
    else:
        print("CPU\n")
        with tf.Session() as sess:
            model = denoiser(sess, sigma=args.sigma)
            if args.phase == 'train':
                denoiser_train(model, lr=lr)
            elif args.phase == 'test':
                denoiser_test(model)
            else:
                print('[!]Unknown phase')
                exit(0)


if __name__ == '__main__':
    tf.app.run()
